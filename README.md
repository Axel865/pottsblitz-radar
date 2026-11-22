# pottsblitz-radar

Rain-radar tile renderer for the weather app PottsBlitz (https://www.emscherwetter.de).

Every 5 minutes a GitHub Actions run turns the DWD radar composite (`WN`, from
[opendata.dwd.de](https://opendata.dwd.de/weather/radar/composite/wn/)) into
standard XYZ map tiles (256×256 PNG, 5 zoom levels, 49 time steps from −2 h
to +2 h), colours snow separately from rain using DWD's ICON-D2 model
(2 m temperature and dew point), and uploads them to the app's web space.

- `deploy/radar-tiles/render_tiles.py` – the renderer
- `.github/workflows/radar-tiles.yml` – the workflow (started via
  `workflow_dispatch` by the web server)

This is a separate public repository only because GitHub Actions is free and
unlimited for public repositories. It contains no credentials; they are stored
as repository secrets.

Data: © Deutscher Wetterdienst (DWD), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
