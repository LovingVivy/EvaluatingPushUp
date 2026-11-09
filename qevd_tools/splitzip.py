"""Doc file zip chia nhieu phan (.z01, .z02, ..., .zip) ma khong can ghep hay giai nen het."""
import os, struct, zlib


class SplitZip:
    def __init__(self, parts):
        self.parts = parts                                  # theo thu tu dia: z01, z02, ..., zip
        self.sizes = [os.path.getsize(p) for p in parts]
        self.starts = [sum(self.sizes[:i]) for i in range(len(parts))]
        self.files = [open(p, "rb") for p in parts]
        self.entries = self._read_central_directory()

    def read_at(self, disk, offset, n):
        """Doc n byte bat dau tu (dia, offset), tu dong sang dia ke tiep."""
        out = bytearray()
        while n > 0:
            if offset >= self.sizes[disk]:
                offset -= self.sizes[disk]; disk += 1
                continue
            f = self.files[disk]
            f.seek(offset)
            chunk = f.read(min(n, self.sizes[disk] - offset))
            out += chunk; n -= len(chunk); offset += len(chunk)
        return bytes(out)

    def _read_central_directory(self):
        last = len(self.parts) - 1
        tail_len = min(self.sizes[last], 1 << 17)
        tail = self.read_at(last, self.sizes[last] - tail_len, tail_len)
        pos = tail.rfind(b"PK\x05\x06")
        assert pos >= 0, "Khong thay EOCD"
        disk_no, cd_disk, n_here, n_total, cd_size, cd_off = struct.unpack("<HHHHII", tail[pos + 4:pos + 20])
        loc = tail.rfind(b"PK\x06\x07", 0, pos)
        if loc >= 0:                                        # ZIP64
            z_disk, z_off, _ = struct.unpack("<IQI", tail[loc + 4:loc + 20])
            rec = self.read_at(z_disk, z_off, 56)
            assert rec[:4] == b"PK\x06\x06"
            _, _, _, _, cd_disk, _, n_total, cd_size, cd_off = struct.unpack("<QHHIIQQQQ", rec[4:56])
        cd = self.read_at(cd_disk, cd_off, cd_size)
        entries, p = {}, 0
        while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            (method, _, _, _, csize, usize, nlen, elen, clen, disk, _, _, off) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            q = 0
            while q + 4 <= len(extra):
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    body, b = extra[q + 4:q + 4 + hlen], 0
                    if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if off == 0xFFFFFFFF: off = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if disk == 0xFFFF: disk = struct.unpack("<I", body[b:b + 4])[0]; b += 4
                q += 4 + hlen
            entries[name] = (method, csize, usize, disk, off)
            p += 46 + nlen + elen + clen
        assert len(entries) == n_total or n_total in (0xFFFF,), (len(entries), n_total)
        return entries

    def read(self, name):
        method, csize, usize, disk, off = self.entries[name]
        head = self.read_at(disk, off, 30)
        assert head[:4] == b"PK\x03\x04", f"Sai local header: {name}"
        nlen, elen = struct.unpack("<HH", head[26:30])
        data = self.read_at(disk, off + 30 + nlen + elen, csize)
        if method == 0:
            return data
        assert method == 8, f"Phuong thuc nen la: {method}"
        out = zlib.decompress(data, -15)
        assert len(out) == usize
        return out

    def close(self):
        for f in self.files:
            f.close()
