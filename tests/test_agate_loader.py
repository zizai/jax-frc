from pathlib import Path

from validation.utils.agate_data import AgateDataLoader


def test_agate_loader_cache_layout(tmp_path, monkeypatch):
    loader = AgateDataLoader(cache_dir=tmp_path)
    record = {
        "files": [
            {
                "key": "GEM_data/hallGEM512.grid.h5",
                "links": {"self": "https://example/grid.h5"},
            },
            {
                "key": "GEM_data/hallGEM512.state_0000.h5",
                "links": {"self": "https://example/state0.h5"},
            },
            {
                "key": "hall_OT_data/hallOT256.grid.h5",
                "links": {"self": "https://example/ot_grid.h5"},
            },
        ]
    }
    monkeypatch.setattr(loader, "_fetch_record", lambda: record)

    gem_files = loader._select_files("gem", 512)
    assert all("hallGEM512" in f["key"] for f in gem_files)

    ot_files = loader._select_files("ot", 256)
    assert all("hallOT256" in f["key"] for f in ot_files)

    path = loader._target_path("gem", 512, "hallGEM512.grid.h5")
    assert path == tmp_path / "gem" / "512" / "hallGEM512.grid.h5"


def test_agate_loader_ensure_files_creates_cache(tmp_path, monkeypatch):
    loader = AgateDataLoader(cache_dir=tmp_path)
    record = {
        "files": [
            {
                "key": "GEM_data/hallGEM512.grid.h5",
                "links": {"self": "https://example/grid.h5"},
            }
        ]
    }
    monkeypatch.setattr(loader, "_fetch_record", lambda: record)

    def fake_download(url: str, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"stub")
        return dest

    monkeypatch.setattr(loader, "_download_file", fake_download)
    paths = loader.ensure_files("gem", 512)
    assert paths
    assert paths[0].exists()
