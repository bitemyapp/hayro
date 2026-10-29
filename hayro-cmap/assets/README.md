See https://github.com/LaurenzV/cmap-resources/tree/hayro for information on how this blob was generated and how
it can be parsed. `cmaps.bin` is the decompressed `cmaps.brotli` from there.

`cmap-tables.bin` holds the same cmaps decoded, as tables that lookups search in place (the
format is described in `src/prebuilt.rs`), and is what's embedded. The
`tables_are_the_bundle_decoded` test checks it against `cmaps.bin`; run it with
`HAYRO_WRITE_CMAP_TABLES=1` to write it again after `cmaps.bin` changes.
