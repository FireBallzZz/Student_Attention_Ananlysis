import io
import zipfile
import requests

class HttpRangeFile(io.RawIOBase):
    def __init__(self, url, headers=None):
        self.headers = headers or {"User-Agent": "Mozilla/5.0"}
        self.session = requests.Session()
        r = self.session.head(url, headers=self.headers, allow_redirects=True)
        self.url = r.url
        self.length = int(r.headers.get("Content-Length", 0))
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.length + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size == -1 or self.pos + size > self.length:
            size = self.length - self.pos
        if size <= 0:
            return b""
        end = self.pos + size - 1
        headers = dict(self.headers)
        headers["Range"] = f"bytes={self.pos}-{end}"
        r = self.session.get(self.url, headers=headers)
        if r.status_code in (200, 206):
            data = r.content
            self.pos += len(data)
            return data
        raise IOError(f"HTTP error {r.status_code}")
