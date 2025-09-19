# TouchPros Scraper

This repository contains a Python CLI utility that mirrors the navigation of a
TouchPros-powered site. The workflow is split into two explicit stages so that a
crawl can be resumed if it is interrupted:

1. **`build-sitemap`** walks the developer API for a school domain, following
   every link that resolves to another API endpoint. Discovered pages are stored
   in `sitemap.json`, which is rewritten frequently to capture progress.
2. **`fetch-content`** loads the generated sitemap and downloads JSON/HTML
   payloads plus all referenced assets (images, PDFs, flipbooks, and YouTube
   videos via `yt-dlp`) into a directory tree that matches the site's
   navigation.

## Usage

```bash
# Stage 1: enumerate every page and write sitemap.json
python touchpros_scraper.py build-sitemap tjhsst.touchpros.com

# Stage 2: download content using the saved sitemap
python touchpros_scraper.py fetch-content tjhsst.touchpros.com
```

Useful options:

- `--output PATH` – change the destination directory (default: `./touchpros_dump`).
- `--delay SECONDS` – throttle requests by the specified delay between calls.
- `--max-pages N` – (stage 1) optional safety limit for the number of pages to visit.
- `--write-interval N` – rewrite the sitemap or manifest after every N pages (default: 5).
- `--sitemap FILE` – override the sitemap path (both stages).
- `--manifest FILE` – (stage 2) override the manifest path.
- `--limit N` – (stage 2) only download the first N entries from the sitemap.
- `--verbose` – enable debug logging for deeper insight into the crawl.

The scraper starts at `/Home2`, follows navigation links discovered in the API
responses (including search tabs, profiles, flipbooks, etc.), and records the
slug/hash-based directory name for each page inside the sitemap. Stage two uses
those same identifiers to persist JSON responses, HTML pages, binary assets, and
YouTube videos (downloaded through `yt-dlp`) under `_assets/<host>/...`.

## Notes

- The sitemap format is versioned so subsequent fetches can validate the domain
  before downloading content.
- Results are grouped by navigation hierarchy. Every page has its own folder
  whose name combines a slugified label and a short hash of the API path.
- All assets are stored once under `_assets/<host>/...` and referenced from the
  generated `content_manifest.json` file.
- Each page directory also contains symlinks to the assets it references so
  they are easy to discover when browsing the exported folder tree.
- The tool respects the TouchPros JSON API but gracefully falls back to parsing
  HTML fragments when the API returns pre-rendered markup.
