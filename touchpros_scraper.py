#!/usr/bin/env python3
"""TouchPros content scraper.

This CLI utility operates in two stages:

1. ``build-sitemap`` crawls a TouchPros domain, following developer API
   endpoints to enumerate every reachable page. The discovered structure is
   written to ``sitemap.json`` periodically so progress can be resumed.
2. ``fetch-content`` reads the generated sitemap and downloads the JSON/HTML
   payloads plus referenced assets (images, documents, YouTube videos, etc.).

The output mirrors the site's navigation hierarchy and stores binary assets in
an ``_assets`` directory rooted at the export folder.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple


LINK_TAGS = {"a", "area"}
RESOURCE_ATTRS = {"src", "data-src", "data-original", "poster", "href"}
STYLE_URL_RE = re.compile(r"url\(([^)]+)\)")
ABSOLUTE_URL_RE = re.compile(r"https?://[\w\-./?%&#=:+]+", re.IGNORECASE)
HASH_LENGTH = 10
SITEMAP_VERSION = 1


@dataclass
class Page:
    """Represents a queued page to be fetched from the API."""

    path: str
    label: str
    breadcrumbs: List[str] = field(default_factory=list)
    source: str = ""
    dir_name: Optional[str] = None


@dataclass
class SiteMapEntry:
    """Serializable representation of a sitemap node."""

    path: str
    label: str
    api_url: str
    page_type: str
    dir_name: str
    breadcrumbs: List[str] = field(default_factory=list)
    source: str = ""
    children: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "api_url": self.api_url,
            "page_type": self.page_type,
            "dir_name": self.dir_name,
            "breadcrumbs": list(self.breadcrumbs),
            "source": self.source,
            "children": list(self.children),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiteMapEntry":
        return cls(
            path=data["path"],
            label=data.get("label", data["path"]),
            api_url=data.get("api_url", ""),
            page_type=data.get("page_type", "json"),
            dir_name=data.get("dir_name", slugify(data.get("label", data["path"]))),
            breadcrumbs=data.get("breadcrumbs", []),
            source=data.get("source", ""),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def relative_parts(self) -> List[str]:
        return [*self.breadcrumbs, self.dir_name]


@dataclass
class PageData:
    """Holds the parsed response details for a page."""

    page: Page
    api_url: str
    kind: str
    payload: Optional[Dict[str, Any]]
    text: Optional[str]
    links: List[Tuple[str, str]]
    resource_urls: Set[str]
    gallery_items: List[Dict[str, Any]]
    tw5host: Optional[str]


class LinkAndResourceParser(HTMLParser):
    """Collects links and resource URLs from an HTML fragment."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self.resources: Set[str] = set()
        self._anchor_href: Optional[str] = None
        self._anchor_text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_dict = {name: value for name, value in attrs if value is not None}

        if tag in LINK_TAGS:
            href = attr_dict.get("href")
            if href:
                self._anchor_href = href
                self._anchor_text_parts = []
                if self._looks_like_resource(href):
                    self.resources.add(href)
        else:
            for attr_name in RESOURCE_ATTRS:
                value = attr_dict.get(attr_name)
                if value:
                    self.resources.add(value)

        style = attr_dict.get("style")
        if style:
            for match in STYLE_URL_RE.findall(style):
                cleaned = match.strip("'\"")
                if cleaned:
                    self.resources.add(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag in LINK_TAGS and self._anchor_href:
            text = unescape("".join(self._anchor_text_parts)).strip()
            self.links.append((self._anchor_href, text))
            self._anchor_href = None
            self._anchor_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._anchor_href is not None:
            self._anchor_text_parts.append(data)

    @staticmethod
    def _looks_like_resource(href: str) -> bool:
        lower = href.lower()
        if lower.startswith("javascript:"):
            return False
        if any(
            lower.endswith(ext)
            for ext in (".jpg", ".jpeg", ".png", ".gif", ".mp4", ".pdf", ".mov", ".mp3", ".wav")
        ):
            return True
        return False


def slugify(value: str) -> str:
    value = unescape(value or "").strip()
    if not value:
        return "item"
    value = value.replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9\-_. ]+", "", value)
    value = value.strip().replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value or "item"


class TouchProsScraper:
    """Scrapes the TouchPros developer API for a single domain."""

    def __init__(
        self,
        domain: str,
        output_dir: Path,
        delay: float = 0.0,
        max_pages: Optional[int] = None,
    ) -> None:
        self.domain = domain.rstrip("/")
        self.base_url = f"https://{self.domain}"
        self.api_base = f"{self.base_url}/api/developer"
        self.delay = delay
        self.max_pages = max_pages
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.asset_root = self.output_dir / "_assets"
        self.asset_root.mkdir(exist_ok=True)

        self.queue: Deque[Page] = collections.deque()
        self.visited: Set[str] = set()
        self.asset_cache: Dict[str, Path] = {}
        self.sitemap_entries: List[SiteMapEntry] = []
        self.manifest_entries: List[Dict[str, Any]] = []

        self.user_agent = "TouchProsScraper/2.0"

    # ------------------------------------------------------------------
    # Stage 1: Sitemap construction
    # ------------------------------------------------------------------
    def build_sitemap(self, sitemap_path: Path, write_interval: int = 5) -> None:
        logging.info("Starting sitemap build for %s", self.domain)
        write_interval = max(1, write_interval)
        sitemap_path.parent.mkdir(parents=True, exist_ok=True)

        self.queue.clear()
        self.visited.clear()
        self.sitemap_entries = []

        self.enqueue(Page(path="Home2", label="Home"))
        processed = 0
        since_write = 0

        while self.queue:
            if self.max_pages is not None and processed >= self.max_pages:
                logging.info("Reached max pages limit (%s)", self.max_pages)
                break

            page = self.queue.popleft()
            if page.path in self.visited:
                continue

            try:
                page_data = self._fetch_page_data(page)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to process %s: %s", page.path, exc)
                continue

            dir_name = self._compute_dir_name(page)
            page.dir_name = dir_name

            entry = SiteMapEntry(
                path=page.path,
                label=page.label,
                api_url=page_data.api_url,
                page_type=page_data.kind,
                dir_name=dir_name,
                breadcrumbs=list(page.breadcrumbs),
                source=page.source,
                metadata={"title": page.label},
            )

            children: List[Dict[str, str]] = []
            for href, text in page_data.links:
                normalized = self._normalize_path(href)
                if not normalized or not self._is_page_path(normalized):
                    continue
                child_label = text or normalized
                child = Page(
                    path=normalized,
                    label=child_label,
                    breadcrumbs=page.breadcrumbs + [dir_name],
                    source=page.path,
                )
                self.enqueue(child)
                children.append({"path": normalized, "label": child_label})
            entry.children = children

            self.sitemap_entries.append(entry)
            self.visited.add(page.path)
            processed += 1
            since_write += 1

            if (
                processed == 1
                or processed % write_interval == 0
                or not self.queue
            ):
                logging.info(
                    "Sitemap progress: processed %s pages (last: %s), %s remaining in queue",
                    processed,
                    page.path,
                    len(self.queue),
                )

            if since_write >= write_interval:
                self._write_sitemap(sitemap_path)
                since_write = 0

            if self.delay:
                time.sleep(self.delay)

        self._write_sitemap(sitemap_path)
        logging.info("Sitemap completed with %s entries", len(self.sitemap_entries))

    # ------------------------------------------------------------------
    # Stage 2: Content fetching
    # ------------------------------------------------------------------
    def fetch_content(
        self,
        sitemap_path: Path,
        manifest_path: Path,
        write_interval: int = 5,
        limit: Optional[int] = None,
    ) -> None:
        logging.info("Starting content fetch for %s", self.domain)
        write_interval = max(1, write_interval)

        entries = self._load_sitemap(sitemap_path)
        if limit is not None:
            entries = entries[:limit]

        self.manifest_entries = []
        processed = 0
        since_write = 0
        total_entries = len(entries)

        for entry in entries:
            page = Page(
                path=entry.path,
                label=entry.label,
                breadcrumbs=list(entry.breadcrumbs),
                source=entry.source,
                dir_name=entry.dir_name,
            )
            page_dir = self.output_dir.joinpath(*entry.relative_parts)
            page_dir.mkdir(parents=True, exist_ok=True)

            try:
                page_data = self._fetch_page_data(page)
            except Exception as exc:  # pylint: disable=broad-except
                logging.error("Failed to download %s: %s", page.path, exc)
                continue

            manifest_entry = self._write_page_content(page_data, page_dir)
            self.manifest_entries.append(manifest_entry)
            processed += 1
            since_write += 1

            if (
                processed == 1
                or processed == total_entries
                or processed % write_interval == 0
            ):
                logging.info(
                    "Content fetch progress: downloaded %s of %s pages (last: %s)",
                    processed,
                    total_entries,
                    entry.path,
                )

            if since_write >= write_interval:
                self._write_manifest(manifest_path)
                since_write = 0

            if self.delay:
                time.sleep(self.delay)

        self._write_manifest(manifest_path)
        logging.info("Fetched content for %s pages", processed)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def _fetch_page_data(self, page: Page) -> PageData:
        api_url = self._build_api_url(page.path)
        content, content_type = self._fetch(api_url)
        base_view_url = self._view_url(page.path)

        is_json = False
        payload: Optional[Dict[str, Any]] = None
        text: Optional[str] = None

        if content_type and "json" in content_type.lower():
            is_json = True
        elif content.strip().startswith((b"{", b"[")):
            is_json = True

        links: List[Tuple[str, str]] = []
        resource_urls: Set[str] = set()
        gallery_items: List[Dict[str, Any]] = []
        tw5host: Optional[str] = None

        if is_json:
            try:
                payload = json.loads(content.decode("utf-8"))
            except json.JSONDecodeError:
                logging.warning("Response for %s was not valid JSON; treating as HTML", page.path)
                text = content.decode("utf-8", errors="replace")
            else:
                label = payload.get("tabname") or payload.get("title") or page.label or page.path
                page.label = str(label)
                tw5host = str(payload.get("tw5host", "")) or None

                for key in ("nav", "mobileNav", "filter", "descriptor", "eventHtml"):
                    html_fragment = payload.get(key)
                    if isinstance(html_fragment, str) and html_fragment:
                        fragment_links, fragment_resources = self._parse_html_fragment(html_fragment, base_view_url)
                        links.extend(fragment_links)
                        resource_urls.update(fragment_resources)

                script_json = payload.get("scriptJson")
                if isinstance(script_json, str) and script_json:
                    try:
                        parsed = json.loads(script_json)
                        script_urls = self._extract_urls_from_object(parsed)
                        for url in script_urls:
                            resolved = self._resolve_url(url, base_url=base_view_url)
                            if resolved:
                                resource_urls.add(resolved)
                    except json.JSONDecodeError:
                        logging.debug("Unable to decode scriptJson for %s", page.path)

                if isinstance(payload.get("sliderImages"), list) and tw5host:
                    for entry in payload["sliderImages"]:
                        if isinstance(entry, dict) and entry.get("image"):
                            resource_urls.add(f"{tw5host}/Slides/{entry['image']}")

                if isinstance(payload.get("carouselImages"), list) and tw5host:
                    for entry in payload["carouselImages"]:
                        if isinstance(entry, dict) and entry.get("image"):
                            resource_urls.add(urllib.parse.urljoin(tw5host + "/", entry["image"]))

                gallery_json = payload.get("galleryJson")
                if isinstance(gallery_json, str) and gallery_json:
                    try:
                        parsed_gallery = json.loads(gallery_json)
                        if isinstance(parsed_gallery, list):
                            gallery_items = [item for item in parsed_gallery if isinstance(item, dict)]
                    except json.JSONDecodeError:
                        logging.debug("galleryJson for %s is not valid JSON", page.path)
        if payload is None and text is None:
            text = content.decode("utf-8", errors="replace")

        if text is not None and not payload:
            if not page.label:
                page.label = page.path
            html_links, html_resources = self._parse_html_fragment(text, base_view_url)
            links.extend(html_links)
            resource_urls.update(html_resources)

        kind = "json" if payload is not None else "html"
        return PageData(
            page=page,
            api_url=api_url,
            kind=kind,
            payload=payload,
            text=text,
            links=links,
            resource_urls=resource_urls,
            gallery_items=gallery_items,
            tw5host=tw5host,
        )

    def _write_page_content(self, page_data: PageData, page_dir: Path) -> Dict[str, Any]:
        base_entry: Dict[str, Any] = {
            "path": page_data.page.path,
            "label": page_data.page.label,
            "dir": str(page_dir.relative_to(self.output_dir)),
            "api_url": page_data.api_url,
            "type": page_data.kind,
        }

        if page_data.kind == "json" and page_data.payload is not None:
            content_entry = self._write_json_content(page_data, page_dir)
        else:
            content_entry = self._write_html_content(page_data, page_dir)

        base_entry.update(content_entry)
        return base_entry

    def _write_json_content(self, page_data: PageData, page_dir: Path) -> Dict[str, Any]:
        json_path = page_dir / "data.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(page_data.payload or {}, handle, indent=2, ensure_ascii=False)

        assets = self._deduplicate_assets(
            self._download_resources(page_data.resource_urls)
            + self._download_gallery_items(page_data)
        )

        return {
            "data_file": str(json_path.relative_to(self.output_dir)),
            "assets": assets,
        }

    def _write_html_content(self, page_data: PageData, page_dir: Path) -> Dict[str, Any]:
        html_path = page_dir / "page.html"
        html_path.write_text(page_data.text or "", encoding="utf-8")

        assets = self._deduplicate_assets(self._download_resources(page_data.resource_urls))

        return {
            "data_file": str(html_path.relative_to(self.output_dir)),
            "assets": assets,
        }

    def _download_resources(self, urls: Iterable[str]) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for url in sorted(set(urls)):
            local = self._download_asset(url)
            if local:
                results.append(
                    {
                        "source": url,
                        "path": str(local.relative_to(self.output_dir)),
                        "kind": "binary",
                    }
                )
        return results

    def _download_gallery_items(self, page_data: PageData) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for item in page_data.gallery_items:
            image_val = str(item.get("Image", "")) if item.get("Image") else ""
            if not image_val:
                continue

            if item.get("IsVideo"):
                source_url = image_val
                local = self._download_video_asset(source_url)
                kind = "video"
            else:
                source_url = self._resolve_gallery_image_url(image_val, page_data)
                if not source_url:
                    continue
                local = self._download_asset(source_url)
                kind = "binary"

            if local:
                results.append(
                    {
                        "source": source_url,
                        "path": str(local.relative_to(self.output_dir)),
                        "kind": kind,
                    }
                )
        return results

    def _resolve_gallery_image_url(self, image_val: str, page_data: PageData) -> Optional[str]:
        if not image_val:
            return None
        if image_val.startswith("http"):
            return image_val
        if "/" not in image_val and page_data.tw5host:
            # The image path is probably provided elsewhere; avoid guessing paths.
            return None
        base = page_data.tw5host.rstrip("/") + "/" if page_data.tw5host else self._view_url(page_data.page.path)
        return urllib.parse.urljoin(base, image_val)

    @staticmethod
    def _deduplicate_assets(assets: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        seen: Set[Tuple[str, str]] = set()
        deduped: List[Dict[str, str]] = []
        for item in assets:
            key = (item.get("source", ""), item.get("path", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _compute_dir_name(self, page: Page) -> str:
        label = page.label or page.path
        safe_label = slugify(label)
        digest = hashlib.sha1(page.path.encode("utf-8")).hexdigest()[:HASH_LENGTH]
        return f"{safe_label}-{digest}"

    def _write_sitemap(self, sitemap_path: Path) -> None:
        data = {
            "version": SITEMAP_VERSION,
            "domain": self.domain,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self.sitemap_entries),
            "entries": [entry.to_dict() for entry in self.sitemap_entries],
        }
        temp_path = sitemap_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        temp_path.replace(sitemap_path)
        logging.debug("Wrote sitemap to %s", sitemap_path)

    def _load_sitemap(self, sitemap_path: Path) -> List[SiteMapEntry]:
        if not sitemap_path.exists():
            raise FileNotFoundError(f"Sitemap not found: {sitemap_path}")
        with sitemap_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            entries_data = data.get("entries", [])
            domain = data.get("domain")
            if domain and domain != self.domain:
                logging.warning("Sitemap domain %s does not match requested domain %s", domain, self.domain)
        else:
            entries_data = data
        return [SiteMapEntry.from_dict(item) for item in entries_data]

    def _write_manifest(self, manifest_path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domain": self.domain,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(self.manifest_entries),
            "entries": self.manifest_entries,
        }
        temp_path = manifest_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        temp_path.replace(manifest_path)
        logging.debug("Wrote manifest to %s", manifest_path)

    # ------------------------------------------------------------------
    # Networking helpers
    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> Tuple[bytes, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/plain, text/html;q=0.8,*/*;q=0.5",
            "X-Requested-With": "XMLHttpRequest",
        }
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            logging.error("HTTP error for %s: %s", url, exc)
            raise
        except urllib.error.URLError as exc:
            logging.error("Network error for %s: %s", url, exc)
            raise
        return content, content_type

    def _download_asset(self, url: str) -> Optional[Path]:
        if not url:
            return None
        if url in self.asset_cache:
            return self.asset_cache[url]

        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme.startswith("http"):
            return None

        relative_path = parsed.path.lstrip("/") or "index.html"
        target = self.asset_root / parsed.netloc / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            self.asset_cache[url] = target
            return target

        try:
            content, _ = self._fetch(url)
        except Exception:  # pylint: disable=broad-except
            return None

        with target.open("wb") as handle:
            handle.write(content)
        self.asset_cache[url] = target
        return target

    def _download_video_asset(self, url: str) -> Optional[Path]:
        if not url:
            return None
        if url in self.asset_cache:
            return self.asset_cache[url]

        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc or "videos"
        target_dir = self.asset_root / netloc
        target_dir.mkdir(parents=True, exist_ok=True)

        video_id = self._extract_youtube_id(url)
        if video_id:
            existing = self._find_existing_video_file(target_dir, video_id)
            if existing:
                self.asset_cache[url] = existing
                return existing

        before_files = self._list_video_files(target_dir)
        command = [
            "yt-dlp",
            "--no-progress",
            "--restrict-filenames",
            "--no-overwrites",
            "-o",
            str(target_dir / "%(id)s.%(ext)s"),
            url,
        ]

        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            logging.error("yt-dlp is not available to download video %s", url)
            return None

        if result.returncode != 0:
            stderr = result.stderr.strip() or result.stdout.strip()
            logging.error("yt-dlp failed for %s: %s", url, stderr)
            return None

        after_files = self._list_video_files(target_dir)
        new_files = after_files - before_files
        selected = self._select_new_file(new_files)

        if not selected and video_id:
            selected = self._find_existing_video_file(target_dir, video_id)

        if not selected:
            logging.warning("yt-dlp completed but no video file found for %s", url)
            return None

        self.asset_cache[url] = selected
        return selected

    @staticmethod
    def _list_video_files(directory: Path) -> Set[Path]:
        if not directory.exists():
            return set()
        results = set()
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix in {".json", ".info.json", ".ytdl", ".description", ".part"}:
                continue
            results.add(path)
        return results

    @staticmethod
    def _select_new_file(files: Set[Path]) -> Optional[Path]:
        if not files:
            return None
        return max(files, key=lambda item: item.stat().st_mtime)

    @staticmethod
    def _find_existing_video_file(directory: Path, video_id: str) -> Optional[Path]:
        candidates = sorted(directory.glob(f"{video_id}.*"))
        for path in candidates:
            if path.suffix in {".json", ".info.json", ".ytdl", ".description", ".part"}:
                continue
            if path.is_file():
                return path
        return None

    @staticmethod
    def _extract_youtube_id(url: str) -> Optional[str]:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if "youtube" not in netloc and "youtu.be" not in netloc:
            return None
        if netloc.endswith("youtu.be"):
            return parsed.path.lstrip("/") or None
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[-1]
        params = urllib.parse.parse_qs(parsed.query)
        if "v" in params and params["v"]:
            return params["v"][0]
        return None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parse_html_fragment(self, html_fragment: str, base_url: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
        parser = LinkAndResourceParser()
        parser.feed(html_fragment)
        links = parser.links
        resources = set(parser.resources)

        for match in ABSOLUTE_URL_RE.findall(html_fragment):
            resources.add(match)

        normalized_links = []
        for href, text in links:
            resolved = urllib.parse.urljoin(base_url, href)
            normalized_links.append((resolved, text))

        resolved_resources = set()
        for resource in resources:
            resolved_resource = self._resolve_url(resource, base_url=base_url)
            if resolved_resource:
                resolved_resources.add(resolved_resource)

        return normalized_links, resolved_resources

    def _extract_urls_from_object(self, obj: Any) -> Set[str]:
        results: Set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, str):
                for url in ABSOLUTE_URL_RE.findall(value):
                    results.add(url)
                if "." in value and not value.startswith("http"):
                    results.add(value)
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(obj)
        return results

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def enqueue(self, page: Page) -> None:
        normalized = self._normalize_path(page.path)
        if not normalized:
            return
        page.path = normalized
        if page.path in self.visited or any(existing.path == page.path for existing in self.queue):
            return
        self.queue.append(page)

    def _normalize_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        path = path.strip()
        if not path or path in {"#", "javascript:void(0)"}:
            return None
        if path.startswith("javascript:") or path.startswith("mailto:"):
            return None
        if path.startswith("//"):
            path = "https:" + path
        if path.startswith("http"):
            parsed = urllib.parse.urlparse(path)
            if parsed.netloc.lower() != self.domain.lower():
                return None
            normalized = parsed.path.lstrip("/")
            if parsed.query:
                normalized += "?" + parsed.query
            return normalized
        return path.lstrip("/")

    def _is_page_path(self, path: str) -> bool:
        base = path.split("?", 1)[0]
        name = base.rsplit("/", 1)[-1]
        return "." not in name

    def _build_api_url(self, path: str) -> str:
        if path.lower().startswith("api/"):
            return urllib.parse.urljoin(self.base_url + "/", path)
        return urllib.parse.urljoin(self.api_base + "/", path)

    def _view_url(self, path: str) -> str:
        return urllib.parse.urljoin(self.base_url + "/", path)

    def _resolve_url(self, url: str, base_url: str) -> Optional[str]:
        if not url:
            return None
        url = url.strip()
        if not url or url.startswith("javascript:") or url.startswith("mailto:"):
            return None
        if url.startswith("data:"):
            return None
        resolved = urllib.parse.urljoin(base_url, url)
        return resolved


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download content from a TouchPros domain")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    sitemap_parser = subparsers.add_parser("build-sitemap", help="Crawl the domain and write a sitemap.json file")
    sitemap_parser.add_argument("domain", help="Domain to crawl, e.g. tjhsst.touchpros.com")
    sitemap_parser.add_argument(
        "--output",
        type=Path,
        default=Path("touchpros_dump"),
        help="Directory where the exported data will be written",
    )
    sitemap_parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds")
    sitemap_parser.add_argument("--max-pages", type=int, default=None, help="Optional limit on the number of pages to discover")
    sitemap_parser.add_argument(
        "--write-interval",
        type=int,
        default=5,
        help="Write the sitemap to disk after this many pages",
    )
    sitemap_parser.add_argument(
        "--sitemap",
        type=Path,
        default=None,
        help="Explicit path for the sitemap JSON file (default: <output>/<domain>/sitemap.json)",
    )

    fetch_parser = subparsers.add_parser("fetch-content", help="Download content using an existing sitemap")
    fetch_parser.add_argument("domain", help="Domain to fetch, e.g. tjhsst.touchpros.com")
    fetch_parser.add_argument(
        "--output",
        type=Path,
        default=Path("touchpros_dump"),
        help="Directory where the exported data will be written",
    )
    fetch_parser.add_argument("--delay", type=float, default=0.0, help="Delay between requests in seconds")
    fetch_parser.add_argument(
        "--sitemap",
        type=Path,
        default=None,
        help="Path to the sitemap JSON file (default: <output>/<domain>/sitemap.json)",
    )
    fetch_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Where to write the content manifest JSON (default: <output>/<domain>/content_manifest.json)",
    )
    fetch_parser.add_argument(
        "--write-interval",
        type=int,
        default=5,
        help="Write the manifest to disk after this many pages",
    )
    fetch_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch the first N entries from the sitemap",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.command == "build-sitemap":
        output_dir = args.output / args.domain
        if output_dir.is_dir() and any(output_dir.iterdir()):
            logging.warning("Output directory %s is not empty", output_dir)
        scraper = TouchProsScraper(
            domain=args.domain,
            output_dir=output_dir,
            delay=args.delay,
            max_pages=args.max_pages,
        )
        sitemap_path = args.sitemap or (output_dir / "sitemap.json")
        scraper.build_sitemap(sitemap_path=sitemap_path, write_interval=args.write_interval)
    elif args.command == "fetch-content":
        output_dir = args.output / args.domain
        scraper = TouchProsScraper(
            domain=args.domain,
            output_dir=output_dir,
            delay=args.delay,
        )
        sitemap_path = args.sitemap or (output_dir / "sitemap.json")
        manifest_path = args.manifest or (output_dir / "content_manifest.json")
        scraper.fetch_content(
            sitemap_path=sitemap_path,
            manifest_path=manifest_path,
            write_interval=args.write_interval,
            limit=args.limit,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
