The CMYK profile is from: https://github.com/saucecontrol/Compact-ICC-Profiles?tab=readme-ov-file#cmyk, available under the CC0-1.0 License

The pfb files in this directory were extracted from Pdfium

Original copyright notice:

```
Copyright 2014 PDFium Authors. All rights reserved.
 Use of this source code is governed by a BSD-style license that can be
 found in the LICENSE file.

Original code copyright 2014 Foxit Software Inc. http://www.foxitsoftware.com
```

The LAB.icc profile was generated with LCMS2.
It is available under the CC0-1.0 License.

cmyk-to-srgb.bin is the table moxcms 0.8.1 builds for an 8-bit transform from the CMYK profile
to sRGB with its default options (17^4 grid points, K varying slowest, three levels each, then
one byte of padding), copied out of its fixed-point executor. The `matches_moxcms` test in
`src/color/device_cmyk.rs` checks that it gives what moxcms gives.
