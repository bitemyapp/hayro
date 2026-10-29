//! The predefined cmaps as tables built ahead of time, searched where they are.
//!
//! Decoding a predefined cmap from the compact bundle in `assets/cmaps.bin` took several
//! milliseconds for the large ones (`Adobe-Japan1-UCS2` and the like), more than the rest of
//! a page of CJK text. `assets/cmap-tables.bin` holds the same cmaps decoded, as sorted
//! tables of fixed-size entries that lookups search in place, so that using one costs
//! nothing up front. A test decodes the bundle and checks the tables against it.
//!
//! Each cmap is a record, in the order of [`index`]; all numbers are little-endian:
//!
//! - `hcmap`, the version (1) and the record's length (u32);
//! - flags (u8): 1 if there's a name, 2 a character collection, 4 a cmap it uses, and the
//!   writing mode times 8 (0 for none, 1 horizontal, 2 vertical);
//! - the name, the collection's registry and ordering, and the used cmap's name, those that
//!   are there, each as its length (u8) and bytes, the collection's supplement (i32) after
//!   its ordering;
//! - the number of entries of each table (u32 each): the CID ranges of codes of 1 to 4
//!   bytes, the `.notdef` ranges of codes of 1 to 4 bytes, the bf ranges, and UTF-16 code
//!   units;
//! - the tables, in that order. A CID or `.notdef` range is its first code (u32), its last
//!   less its first (u16) and its first CID (u16); a bf range is its first code (u32), its
//!   last less its first (u16), and where its destination's code units start among the
//!   cmap's (u24) after how many there are (u8). Ranges are sorted and don't overlap.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;
use std::sync::LazyLock;

use crate::{
    CMap, CMapName, CharacterCollection, CidFamily, Metadata, PartitionedRanges, Units,
    WritingMode, parse,
};

pub(crate) const MAGIC: &[u8] = b"hcmap";
const VERSION: u8 = 1;
const HEADER_SIZE: usize = 10;
const CID_ENTRY: usize = 8;
const BF_ENTRY: usize = 10;

static BUNDLE: LazyLock<Bundle> = LazyLock::new(|| {
    let data: &'static [u8] = include_bytes!("../assets/cmap-tables.bin");
    let mut records = Vec::new();
    let mut start = 0;

    while let Some(len) = data.get(start + 6..start + HEADER_SIZE) {
        let len = u32::from_le_bytes(len.try_into().unwrap()) as usize;
        records.push(start..start + len);
        start += len;
    }

    Bundle { data, records }
});

struct Bundle {
    data: &'static [u8],
    records: Vec<Range<usize>>,
}

/// The position of a predefined cmap in the bundles, which hold them in alphabetical order.
pub(crate) fn index(name: CMapName<'_>) -> Option<usize> {
    Some(match name {
        CMapName::N83pvRksjH => 0,
        CMapName::N90msRksjH => 1,
        CMapName::N90msRksjV => 2,
        CMapName::N90mspRksjH => 3,
        CMapName::N90mspRksjV => 4,
        CMapName::N90pvRksjH => 5,
        CMapName::AddRksjH => 6,
        CMapName::AddRksjV => 7,
        CMapName::AdobeCns1Ucs2 => 8,
        CMapName::AdobeGb1Ucs2 => 9,
        CMapName::AdobeJapan1Ucs2 => 10,
        CMapName::AdobeKorea1Ucs2 => 11,
        CMapName::B5pcH => 12,
        CMapName::B5pcV => 13,
        CMapName::CnsEucH => 14,
        CMapName::CnsEucV => 15,
        CMapName::ETenB5H => 16,
        CMapName::ETenB5V => 17,
        CMapName::ETenmsB5H => 18,
        CMapName::ETenmsB5V => 19,
        CMapName::EucH => 20,
        CMapName::EucV => 21,
        CMapName::ExtRksjH => 22,
        CMapName::ExtRksjV => 23,
        CMapName::GbEucH => 24,
        CMapName::GbEucV => 25,
        CMapName::GbkEucH => 26,
        CMapName::GbkEucV => 27,
        CMapName::Gbk2kH => 28,
        CMapName::Gbk2kV => 29,
        CMapName::GbkpEucH => 30,
        CMapName::GbkpEucV => 31,
        CMapName::GbpcEucH => 32,
        CMapName::GbpcEucV => 33,
        CMapName::H => 34,
        CMapName::HKscsB5H => 35,
        CMapName::HKscsB5V => 36,
        CMapName::IdentityH => 37,
        CMapName::IdentityV => 38,
        CMapName::KscEucH => 39,
        CMapName::KscEucV => 40,
        CMapName::KscmsUhcH => 41,
        CMapName::KscmsUhcHwH => 42,
        CMapName::KscmsUhcHwV => 43,
        CMapName::KscmsUhcV => 44,
        CMapName::KscpcEucH => 45,
        CMapName::UniCnsUcs2H => 46,
        CMapName::UniCnsUcs2V => 47,
        CMapName::UniCnsUtf16H => 48,
        CMapName::UniCnsUtf16V => 49,
        CMapName::UniGbUcs2H => 50,
        CMapName::UniGbUcs2V => 51,
        CMapName::UniGbUtf16H => 52,
        CMapName::UniGbUtf16V => 53,
        CMapName::UniJisUcs2H => 54,
        CMapName::UniJisUcs2HwH => 55,
        CMapName::UniJisUcs2HwV => 56,
        CMapName::UniJisUcs2V => 57,
        CMapName::UniJisUtf16H => 58,
        CMapName::UniJisUtf16V => 59,
        CMapName::UniKsUcs2H => 60,
        CMapName::UniKsUcs2V => 61,
        CMapName::UniKsUtf16H => 62,
        CMapName::UniKsUtf16V => 63,
        CMapName::V => 64,
        CMapName::Custom(_) => return None,
    })
}

/// Load the data for a cmap file, by name.
pub fn load_embedded(name: CMapName<'_>) -> Option<&'static [u8]> {
    let range = BUNDLE.records.get(index(name)?)?;

    Some(&BUNDLE.data[range.clone()])
}

/// `bytes` as the bundle's own, if they are (as [`load_embedded`] gives them), so that the
/// tables in them can be searched where they are rather than copied.
fn in_bundle(bytes: &[u8]) -> Option<&'static [u8]> {
    let data = BUNDLE.data;
    let offset = (bytes.as_ptr() as usize).checked_sub(data.as_ptr() as usize)?;

    data.get(offset..offset.checked_add(bytes.len())?)
}

/// The tables of a prebuilt cmap; see the module's documentation.
#[derive(Debug, Clone)]
pub(crate) struct Tables {
    data: Cow<'static, [u8]>,
    cid_ranges: [Range<usize>; 4],
    notdef_ranges: [Range<usize>; 4],
    bf_ranges: Range<usize>,
    units: Range<usize>,
}

impl Tables {
    /// The first code and CID of the range of codes of `byte_len` bytes (1 to 4) holding
    /// `code`, among the CID ranges or the `.notdef` ones.
    pub(crate) fn find_cid(&self, notdef: bool, byte_len: u8, code: u32) -> Option<(u32, u32)> {
        let tables = if notdef {
            &self.notdef_ranges
        } else {
            &self.cid_ranges
        };
        let table = tables.get(usize::from(byte_len).checked_sub(1)?)?;
        let entries = self.data[table.clone()].as_chunks::<CID_ENTRY>().0;
        let entry = search(entries, code, cid_range)?;

        Some((cid_range(entry).0, u32::from(u16_at(entry, 6))))
    }

    /// The first code of the bf range holding `code`, and the first code's destination.
    pub(crate) fn find_bf(&self, code: u32) -> Option<(u32, Units)> {
        let entries = self.data[self.bf_ranges.clone()].as_chunks::<BF_ENTRY>().0;
        let entry = search(entries, code, bf_range)?;
        let count = usize::from(entry[6]);
        let start = u32::from_le_bytes([entry[7], entry[8], entry[9], 0]) as usize;
        let units = self.data[self.units.clone()]
            .as_chunks::<2>()
            .0
            .get(start..start.checked_add(count)?)?;

        Some((
            bf_range(entry).0,
            units.iter().map(|unit| u16::from_le_bytes(*unit)).collect(),
        ))
    }
}

/// The entry whose range holds `code`, found as `crate::find_in_ranges` finds it.
pub(crate) fn search<T>(entries: &[T], code: u32, range: impl Fn(&T) -> (u32, u32)) -> Option<&T> {
    let index = entries
        .binary_search_by(|entry| {
            let (start, end) = range(entry);
            if code < start {
                Ordering::Greater
            } else if code > end {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        })
        .ok()?;

    Some(&entries[index])
}

fn u16_at(entry: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([entry[at], entry[at + 1]])
}

fn code_range(entry: &[u8]) -> (u32, u32) {
    let start = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);

    (start, start.saturating_add(u32::from(u16_at(entry, 4))))
}

fn cid_range(entry: &[u8; CID_ENTRY]) -> (u32, u32) {
    code_range(entry)
}

fn bf_range(entry: &[u8; BF_ENTRY]) -> (u32, u32) {
    code_range(entry)
}

/// Read a prebuilt cmap, with the cmap it uses (through `get_cmap`).
pub(crate) fn parse<'a>(
    data: &[u8],
    get_cmap: impl Fn(CMapName<'_>) -> Option<&'a [u8]> + Clone + 'a,
    depth: u32,
) -> Option<CMap> {
    if data.get(..MAGIC.len())? != MAGIC || *data.get(MAGIC.len())? != VERSION {
        return None;
    }
    let len = u32::from_le_bytes(data.get(6..HEADER_SIZE)?.try_into().ok()?) as usize;
    let record = data.get(..len)?;
    let mut reader = Reader {
        data: record,
        pos: HEADER_SIZE,
    };

    let flags = reader.u8()?;
    let name = if flags & 1 != 0 {
        Some(reader.string()?.to_vec())
    } else {
        None
    };
    let character_collection = if flags & 2 != 0 {
        let registry = reader.string()?;
        let ordering = reader.string()?;
        Some(CharacterCollection {
            family: CidFamily::from_registry_ordering(registry, ordering),
            supplement: i32::from_le_bytes(reader.bytes(4)?.try_into().ok()?),
        })
    } else {
        None
    };
    let base_name = if flags & 4 != 0 {
        Some(reader.string()?)
    } else {
        None
    };
    let writing_mode = match flags >> 3 {
        0 => None,
        1 => Some(WritingMode::Horizontal),
        2 => Some(WritingMode::Vertical),
        _ => return None,
    };

    let mut counts = [0; 10];
    for count in &mut counts {
        *count = u32::from_le_bytes(reader.bytes(4)?.try_into().ok()?) as usize;
    }
    let mut table = |count: usize, size: usize| {
        let start = reader.pos;
        reader.bytes(count.checked_mul(size)?)?;
        Some(start..reader.pos)
    };
    let mut cid_ranges = [0, 1, 2, 3].map(|_| 0..0);
    for (range, &count) in cid_ranges.iter_mut().zip(&counts[..4]) {
        *range = table(count, CID_ENTRY)?;
    }
    let mut notdef_ranges = [0, 1, 2, 3].map(|_| 0..0);
    for (range, &count) in notdef_ranges.iter_mut().zip(&counts[4..8]) {
        *range = table(count, CID_ENTRY)?;
    }
    let bf_ranges = table(counts[8], BF_ENTRY)?;
    let units = table(counts[9], 2)?;
    if reader.pos != record.len() {
        return None;
    }

    let base = match base_name {
        Some(base_name) => Some(Box::new(parse::parse_inner(
            get_cmap(CMapName::from_bytes(base_name))?,
            get_cmap.clone(),
            depth + 1,
        )?)),
        None => None,
    };

    let data = in_bundle(record).map_or_else(|| Cow::Owned(record.to_vec()), Cow::Borrowed);

    Some(CMap {
        metadata: Metadata {
            character_collection,
            name,
            writing_mode,
        },
        _codespace_ranges: Vec::new(),
        cid_ranges: PartitionedRanges::new(),
        notdef_ranges: PartitionedRanges::new(),
        bf_entries: Vec::new(),
        base,
        prebuilt: Some(Box::new(Tables {
            data,
            cid_ranges,
            notdef_ranges,
            bf_ranges,
            units,
        })),
    })
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let bytes = self.data.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;

        Some(bytes)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }

    fn string(&mut self) -> Option<&'a [u8]> {
        let len = usize::from(self.u8()?);

        self.bytes(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BfString, bcmap};

    /// A cmap decoded from the bundle as a prebuilt record.
    fn record(cmap: &CMap) -> Vec<u8> {
        fn string(out: &mut Vec<u8>, bytes: &[u8]) {
            out.push(u8::try_from(bytes.len()).unwrap());
            out.extend_from_slice(bytes);
        }
        fn code_range(out: &mut Vec<u8>, start: u32, end: u32) {
            out.extend_from_slice(&start.to_le_bytes());
            out.extend_from_slice(&u16::try_from(end - start).unwrap().to_le_bytes());
        }

        let metadata = &cmap.metadata;
        let base_name = cmap
            .base
            .as_ref()
            .map(|base| base.metadata.name.clone().unwrap());
        let writing_mode = match metadata.writing_mode {
            None => 0,
            Some(WritingMode::Horizontal) => 1,
            Some(WritingMode::Vertical) => 2,
        };
        let mut out = Vec::from(MAGIC);
        out.push(VERSION);
        out.extend_from_slice(&[0; 4]);
        out.push(
            u8::from(metadata.name.is_some())
                | u8::from(metadata.character_collection.is_some()) << 1
                | u8::from(base_name.is_some()) << 2
                | writing_mode << 3,
        );
        if let Some(name) = &metadata.name {
            string(&mut out, name);
        }
        if let Some(collection) = &metadata.character_collection {
            let (registry, ordering): (&[u8], &[u8]) = match &collection.family {
                CidFamily::AdobeJapan1 => (b"Adobe", b"Japan1"),
                CidFamily::AdobeGB1 => (b"Adobe", b"GB1"),
                CidFamily::AdobeCNS1 => (b"Adobe", b"CNS1"),
                CidFamily::AdobeKorea1 => (b"Adobe", b"Korea1"),
                CidFamily::AdobeIdentity => (b"Adobe", b"Identity"),
                CidFamily::Custom { registry, ordering } => (registry, ordering),
            };
            string(&mut out, registry);
            string(&mut out, ordering);
            out.extend_from_slice(&collection.supplement.to_le_bytes());
        }
        if let Some(base_name) = &base_name {
            string(&mut out, base_name);
        }

        let cid_buckets = [&cmap.cid_ranges, &cmap.notdef_ranges]
            .into_iter()
            .flat_map(|ranges| (1..=4).map(move |n| ranges.get(n).unwrap()))
            .collect::<Vec<_>>();
        let units = cmap
            .bf_entries
            .iter()
            .flat_map(|entry| entry.dst_base.iter().copied())
            .collect::<Vec<_>>();
        let mut counts = cid_buckets.iter().map(|b| b.len()).collect::<Vec<_>>();
        counts.extend([cmap.bf_entries.len(), units.len()]);
        for count in counts {
            out.extend_from_slice(&u32::try_from(count).unwrap().to_le_bytes());
        }

        for bucket in cid_buckets {
            for pair in bucket.windows(2) {
                assert!(pair[0].range.end < pair[1].range.start);
            }
            for range in bucket {
                code_range(&mut out, range.range.start, range.range.end);
                out.extend_from_slice(&u16::try_from(range.cid_start).unwrap().to_le_bytes());
            }
        }
        for pair in cmap.bf_entries.windows(2) {
            assert!(pair[0].range.end < pair[1].range.start);
        }
        let mut start = 0_u32;
        for entry in &cmap.bf_entries {
            code_range(&mut out, entry.range.start, entry.range.end);
            out.push(u8::try_from(entry.dst_base.len()).unwrap());
            assert!(start < 1 << 24);
            out.extend_from_slice(&start.to_le_bytes()[..3]);
            start += u32::try_from(entry.dst_base.len()).unwrap();
        }
        for unit in units {
            out.extend_from_slice(&unit.to_le_bytes());
        }

        let len = u32::try_from(out.len()).unwrap().to_le_bytes();
        out[6..HEADER_SIZE].copy_from_slice(&len);
        out
    }

    /// Every predefined cmap, decoded from the bundle.
    fn decoded() -> Vec<CMap> {
        (0..65)
            .map(|index| bcmap::parse(bcmap::load_index(index).unwrap(), bcmap::load, 0).unwrap())
            .collect()
    }

    /// The tables are the bundle's cmaps decoded. With `HAYRO_WRITE_CMAP_TABLES` set, they're
    /// written from it instead.
    #[test]
    fn tables_are_the_bundle_decoded() {
        let tables = decoded().iter().flat_map(record).collect::<Vec<_>>();

        if std::env::var_os("HAYRO_WRITE_CMAP_TABLES").is_some() {
            std::fs::write(
                concat!(env!("CARGO_MANIFEST_DIR"), "/assets/cmap-tables.bin"),
                &tables,
            )
            .unwrap();
        }

        assert!(
            tables == BUNDLE.data,
            "assets/cmap-tables.bin is out of date"
        );
    }

    /// Each prebuilt cmap gives what it gives decoded, for every code of one and two bytes
    /// and around each range's ends.
    #[test]
    fn tables_look_up_as_decoded() {
        for (index, decoded) in decoded().iter().enumerate() {
            let record = &BUNDLE.data[BUNDLE.records[index].clone()];
            let prebuilt = CMap::parse(record, load_embedded).unwrap();
            assert!(matches!(
                &prebuilt.prebuilt.as_ref().unwrap().data,
                Cow::Borrowed(_)
            ));
            assert_eq!(
                std::format!("{:?}", prebuilt.metadata),
                std::format!("{:?}", decoded.metadata)
            );
            assert_eq!(prebuilt.base.is_some(), decoded.base.is_some());

            let mut codes = (0..=0xFFFF).collect::<Vec<u32>>();
            let mut ends = |cmap: &CMap| {
                for n in 1..=4 {
                    for ranges in [&cmap.cid_ranges, &cmap.notdef_ranges] {
                        for range in ranges.get(n).unwrap() {
                            let (start, end) = (range.range.start, range.range.end);
                            codes.extend([start.wrapping_sub(1), start, end, end.wrapping_add(1)]);
                        }
                    }
                }
                for entry in &cmap.bf_entries {
                    let (start, end) = (entry.range.start, entry.range.end);
                    codes.extend([start.wrapping_sub(1), start, end, end.wrapping_add(1)]);
                }
            };
            ends(decoded);
            if let Some(base) = &decoded.base {
                ends(base);
            }

            for &code in &codes {
                for byte_len in 0..=5 {
                    assert_eq!(
                        prebuilt.lookup_cid_code(code, byte_len),
                        decoded.lookup_cid_code(code, byte_len),
                        "{index}: {code:#x} in {byte_len} bytes"
                    );
                }
                let bf: Option<BfString> = decoded.lookup_bf_string(code);
                assert_eq!(prebuilt.lookup_bf_string(code), bf, "{index}: {code:#x}");
            }
        }
    }

    #[test]
    fn copied_tables_look_up_as_borrowed() {
        let record = load_embedded(CMapName::AdobeJapan1Ucs2).unwrap();
        let copy = record.to_vec();
        let borrowed = CMap::parse(record, load_embedded).unwrap();
        let copied = CMap::parse(&copy, load_embedded).unwrap();
        assert!(matches!(
            &copied.prebuilt.as_ref().unwrap().data,
            Cow::Owned(_)
        ));
        for code in 0..30_000 {
            assert_eq!(
                copied.lookup_bf_string(code),
                borrowed.lookup_bf_string(code)
            );
        }
    }

    #[test]
    fn malformed_records_are_refused() {
        let record = load_embedded(CMapName::UniJisUcs2V).unwrap();
        assert!(CMap::parse(&record[..record.len() - 1], load_embedded).is_none());
        let mut longer = record.to_vec();
        longer.push(0);
        let len = u32::try_from(longer.len()).unwrap().to_le_bytes();
        longer[6..HEADER_SIZE].copy_from_slice(&len);
        assert!(CMap::parse(&longer, load_embedded).is_none());
        let mut versioned = record.to_vec();
        versioned[5] = 2;
        assert!(CMap::parse(&versioned, load_embedded).is_none());
    }
}
