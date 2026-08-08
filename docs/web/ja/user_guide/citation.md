# 引用方法 (Citation)

GWexpy を研究や論文等で使用した場合は、実際に使用した版を以下の形式で引用してください。

## 論文での引用 (BibTeX)

解析結果の再現性とソフトウェアの継続的なメンテナンスのために、以下の BibTeX エントリを使用してください。

```bibtex
@software{gwexpy2026,
  author = {Washimi, Tatsuki},
  title = {GWexpy: Extending GWpy with metadata-preserving multidimensional abstractions for detector commissioning},
  year = {2026},
  url = {https://github.com/tatsuki-washimi/gwexpy},
  version = {<version used>}
}
```

## CITATION.cff

再現可能な引用には、可変な `main` ブランチではなく、使用した正確なリリースタグの
`CITATION.cff` を使用してください（例:
`https://github.com/tatsuki-washimi/gwexpy/blob/<exact tag>/CITATION.cff`）。

## 関連する先行研究

GWexpy は以下のソフトウェアに基づいています。これらについても適切に引用することを推奨します。

* **GWpy**: `Duncan Macleod et al., gwpy/gwpy: ...`
* **Astropy**: `Astropy Collaboration et al., ...`
