"""Turn download listings into plain TSV manifests. Standard library only; reads data, runs nothing.

    python -I scripts/fetch_manifest.py hf <listing.json> <out.tsv>
    python -I scripts/fetch_manifest.py lfs <pointer-dump.txt> <out.tsv>

hf:  a Hugging Face tree listing -> "path<TAB>size<TAB>algo<TAB>digest".
     Files under LFS carry a sha256; plain files only have a git blob sha1.
lfs: "path<TAB>size<TAB>sha256" read from a dump of LFS pointer files, see fetch_external.sh.

Paths come from the network, so anything that could escape the target directory is refused.
"""
import json
import sys


def safe(path):
    parts = path.split("/")
    if not path or path.startswith("/") or ".." in parts or "\\" in path or "\n" in path:
        raise SystemExit("refusing unsafe path in listing: {!r}".format(path))
    return path


def from_hf(listing_path, out_path):
    with open(listing_path) as handle:
        listing = json.load(handle)
    rows = []
    for entry in listing:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs")
        if lfs:
            rows.append((safe(entry["path"]), int(lfs["size"]), "sha256", lfs["oid"]))
        else:
            rows.append((safe(entry["path"]), int(entry["size"]), "gitsha1", entry["oid"]))
    rows.sort()
    with open(out_path, "w") as handle:
        for row in rows:
            handle.write("{}\t{}\t{}\t{}\n".format(*row))
    print("{} files, {} bytes".format(len(rows), sum(r[1] for r in rows)))


def from_pointers(dump_path, out_path):
    """The dump is repeated blocks: "@@ <path>" then the blob text of that path."""
    rows, path, oid, size = [], None, None, None

    def flush():
        if path and oid and size is not None:
            rows.append((safe(path), size, oid))

    with open(dump_path, errors="replace") as handle:
        for line in handle:
            if line.startswith("@@ "):
                flush()
                path, oid, size = line[3:].rstrip("\n"), None, None
            elif line.startswith("oid sha256:"):
                oid = line[len("oid sha256:"):].strip()
            elif line.startswith("size "):
                size = int(line[5:].strip())
    flush()
    rows.sort()
    with open(out_path, "w") as handle:
        for row in rows:
            handle.write("{}\t{}\t{}\n".format(*row))
    print("{} LFS files, {} bytes".format(len(rows), sum(r[1] for r in rows)))


if __name__ == "__main__":
    kind, source, target = sys.argv[1:4]
    if kind == "hf":
        from_hf(source, target)
    elif kind == "lfs":
        from_pointers(source, target)
    else:
        raise SystemExit("unknown kind {!r}".format(kind))
