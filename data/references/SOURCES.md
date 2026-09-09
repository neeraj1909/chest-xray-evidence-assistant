# Reference corpus source ledger

The indexed files in `documents/` are short, manually curated paraphrases of
FDA public-information pages. They contain no patient records, images, copied
logos, diagnostic rules, or treatment recommendations.

All sources were accessed on 2026-09-10:

- `radiography-formation.md` derives from “Description” on the FDA
  [Radiography](https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/radiography)
  page.
- `projection-vs-ct.md` derives from “Conventional X-ray Images” and “Computed
  Tomography (CT)” on the FDA
  [What is Computed Tomography?](https://www.fda.gov/radiation-emitting-products/medical-x-ray-imaging/what-computed-tomography)
  page.
- `radiation-purpose.md` derives from “Description,” “Benefits - Risks,” and
  “Principles of radiation protection” on the FDA
  [Medical X-ray Imaging](https://www.fda.gov/radiation-emitting-products/medical-imaging/medical-x-ray-imaging)
  page.

The FDA [Website Policies](https://www.fda.gov/about-fda/about-website/website-policies)
state that FDA website content is generally public domain unless otherwise
noted. No exception or third-party credit was present in the text sections used
here. These repository-authored paraphrases are recorded as `license_cleared`
so the manifest does not overstate their status as verbatim federal works.

The upstream pages can change. The manifest pins the exact local bytes,
access date, source section, transformation, and extractor version; updates
must create new rendition and corpus fingerprints.
