use std::ops::Range;
use std::sync::LazyLock;

use super::huffman::{self, HuffmanTable};
use super::reader::Reader;
use crate::CMapName;

pub(super) static BUNDLE: LazyLock<Bundle> = LazyLock::new(|| {
    // We already know the bundle is valid, so we can skip validation and just
    // unwrap everywhere. It's stored uncompressed: decompressing it took longer than
    // everything else a page of CJK text needs.

    let data: &'static [u8] = include_bytes!("../../assets/cmaps.bin");
    let mut reader = Reader::new(data);
    let huff_size = reader.read_u32().unwrap() as usize;
    let huff_data = reader.read_bytes(huff_size).unwrap();
    let (delta_table, count_table) = huffman::decode_tables(huff_data).unwrap();

    let mut entries = Vec::new();

    while !reader.at_end() {
        let start = reader.position();

        // Skip file magic and version.
        reader.read_bytes(6).unwrap();
        let file_len = reader.read_u32().unwrap() as usize;

        reader.read_bytes(file_len - 10).unwrap();
        entries.push(start..start + file_len);
    }

    Bundle {
        data,
        delta_table,
        count_table,
        entries,
    }
});

pub(super) struct Bundle {
    data: &'static [u8],
    pub(super) delta_table: HuffmanTable,
    pub(super) count_table: HuffmanTable,
    entries: Vec<Range<usize>>,
}

/// The bundle's data for a cmap, by name.
pub(crate) fn load(name: CMapName<'_>) -> Option<&'static [u8]> {
    load_index(crate::prebuilt::index(name)?)
}

/// The bundle's data for the cmap at `index` (see `prebuilt::index`).
pub(crate) fn load_index(index: usize) -> Option<&'static [u8]> {
    let range = BUNDLE.entries.get(index)?;

    Some(&BUNDLE.data[range.clone()])
}
