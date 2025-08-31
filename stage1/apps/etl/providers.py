from __future__ import annotations

import os, time, math, requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

class OpenAQv3:
    BASE = 'https://api.openaq.org/v3'
    PM25_ID = 2
    CITY_LL = {'berlin': (52.52, 13.405), 'hamburg': (53.55, 9.993), 'munich': (48.137, 11.575)}

    def __init__(self, api_key: Optional[str]=None, timeout:int=30) -> None:
        self.api_key = (api_key or os.environ.get('OPENAQ_API_KEY') or '').strip()
        self.timeout = timeout
        self.sess = requests.Session()
        if self.api_key:
            self.sess.headers.update({'X-API-Key': self.api_key, 'Authorization': f'Bearer {self.api_key}'})
        self.proxies = {'http': os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'),
                        'https': os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')}

    def _get(self, path: str, params: Dict[str, Any], max_retries: int=7) -> Dict[str, Any]:
        url = f'{self.BASE}{path}'
        wait = 1.0
        for _ in range(max_retries):
            resp = self.sess.get(url, params=params, timeout=self.timeout, proxies=self.proxies or None)
            if resp.status_code == 429:
                ra = resp.headers.get('Retry-After')
                try: sleep_s = float(ra) if ra is not None else wait
                except Exception: sleep_s = wait
                time.sleep(max(1.0, min(sleep_s, 30.0)))
                wait = min(wait * 2.0, 30.0)
                continue
            if resp.status_code == 401:
                raise PermissionError('OpenAQ v3 401 Unauthorized')
            resp.raise_for_status()
            return resp.json()
        raise SystemExit(f'OpenAQ v3 still rate-limited after {max_retries} attempts on {url}')

    def _bbox_for_city(self, city: str, pad_deg: float=2.0) -> Optional[str]:
        ll = self.CITY_LL.get(city.lower())
        if not ll: return None
        lat, lon = ll
        return f'{lon-pad_deg},{lat-pad_deg},{lon+pad_deg},{lat+pad_deg}'

    def list_locations_pm25(self, *, iso: Optional[str]=None, bbox: Optional[str]=None) -> List[Dict[str,Any]]:
        params = {'parameters_id': str(self.PM25_ID), 'limit': 1000, 'page': 1}
        if iso: params['iso'] = iso
        if bbox: params['bbox'] = bbox
        out, page = [], 1
        while True:
            params['page'] = page
            data = self._get('/locations', params)
            items = data.get('results') or []
            out.extend(items)
            meta = data.get('meta') or {}
            if not items or (meta.get('page', page) * meta.get('limit', 1000) >= meta.get('found', 0)): break
            page += 1; time.sleep(0.15)
        return out

    def sensors_for_location(self, loc_id:int) -> List[Dict[str,Any]]:
        data = self._get(f'/locations/{loc_id}/sensors', {'limit': 1000, 'page': 1})
        return data.get('results') or []

    def hours_for_sensor(self, sensor_id:int, start:datetime, end:datetime) -> List[Dict[str,Any]]:
        out, page = [], 1
        params = {'datetime_from': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'datetime_to':   end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'limit': 1000, 'page': page}
        while True:
            params['page'] = page
            data = self._get(f'/sensors/{sensor_id}/hours', params)
            items = data.get('results') or []
            out.extend(items)
            meta = data.get('meta') or {}
            if not items or (meta.get('page', page) * meta.get('limit', 1000) >= meta.get('found', 0)): break
            page += 1; time.sleep(0.25)
        return out

    def measurements_for_sensor(self, sensor_id:int, start:datetime, end:datetime) -> List[Dict[str,Any]]:
        out, page = [], 1
        params = {'datetime_from': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'datetime_to':   end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'parameter_id':  self.PM25_ID,
                  'limit': 1000, 'page': page,
                  'sort':'asc','order_by':'datetime'}
        while True:
            params['page'] = page
            data = self._get(f'/sensors/{sensor_id}/measurements', params)
            items = data.get('results') or []
            out.extend(items)
            meta = data.get('meta') or {}
            if not items or (meta.get('page', page) * meta.get('limit', 1000) >= meta.get('found', 0)): break
            page += 1; time.sleep(0.25)
        return out

    def measurements_bbox(self, bbox: str, start:datetime, end:datetime) -> List[Dict[str,Any]]:
        out, page = [], 1
        params = {'parameters_id': str(self.PM25_ID), 'bbox': bbox,
                  'datetime_from': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'datetime_to':   end.strftime('%Y-%m-%dT%H:%M:%SZ'),
                  'limit': 1000, 'page': page,
                  'sort':'asc','order_by':'datetime'}
        while True:
            params['page'] = page
            data = self._get('/measurements', params)
            items = data.get('results') or []
            out.extend(items)
            meta = data.get('meta') or {}
            if not items or (meta.get('page', page) * meta.get('limit', 1000) >= meta.get('found', 0)): break
            page += 1; time.sleep(0.25)
        return out

    def _normalize_hourly(self, series: List[Dict[str,Any]], sensor_id: int) -> List[Dict[str,Any]]:
        rows = []
        for r in series:
            dt = ((r.get('datetimeTo') or {}).get('utc')
               or (r.get('datetimeFrom') or {}).get('utc')
               or (r.get('datetime') or {}).get('utc'))
            val = r.get('avg') if 'avg' in r else r.get('value')
            unit = r.get('unit') or ((r.get('parameter') or {}).get('units')) or 'µg/m³'
            if dt is not None and val is not None:
                rows.append({'datetime': dt, 'value': val, 'unit': unit, 'sensor_id': sensor_id})
        return rows

    def _latest_pm25_sensor_ids_in_bbox(self, bbox:str, since_iso: Optional[str]=None, limit:int=1000) -> List[int]:
        params = {'limit': limit}
        if since_iso: params['datetime_min'] = since_iso
        data = self._get(f'/parameters/{self.PM25_ID}/latest', params)
        results = data.get('results') or []
        xmin, ymin, xmax, ymax = [float(p) for p in bbox.split(',')]
        ids = []
        for it in results:
            coords = it.get('coordinates') or {}
            lat, lon = coords.get('latitude'), coords.get('longitude')
            sid = it.get('sensorsId') or it.get('sensorId')
            if lat is None or lon is None or sid is None: continue
            if (xmin <= float(lon) <= xmax) and (ymin <= float(lat) <= ymax):
                try: ids.append(int(sid))
                except: pass
        return sorted(set(ids))

    def _discover_sensors(self, city:str, iso_hint: Optional[str]) -> List[int]:
        bbox = self._bbox_for_city(city)
        sensor_ids = []
        locs = self.list_locations_pm25(bbox=bbox) if bbox else []
        if not locs and iso_hint:
            locs = self.list_locations_pm25(iso=iso_hint)
        for loc in locs:
            for s in self.sensors_for_location(int(loc['id'])):
                param = (s.get('parameter') or {})
                if int(param.get('id', -1)) == self.PM25_ID:
                    try: sensor_ids.append(int(s['id']))
                    except: pass
        return sorted(set(sensor_ids))

    def fetch_pm25_city(self, *, city:str, hours:int, iso_hint: Optional[str]) -> List[Dict[str,Any]]:
        now_utc = datetime.now(timezone.utc)
        start   = now_utc - timedelta(hours=hours)
        bbox    = self._bbox_for_city(city)
        rows: List[Dict[str,Any]] = []

        candidates = []
        if bbox:
            since30 = (now_utc - timedelta(days=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
            candidates = self._latest_pm25_sensor_ids_in_bbox(bbox, since_iso=since30)
        if not candidates:
            candidates = self._discover_sensors(city, iso_hint)

        MAX_SENSORS = 8
        candidates = candidates[:MAX_SENSORS]

        def pull_for(ids: List[int]) -> List[Dict[str,Any]]:
            acc = []
            for sid in ids:
                series_h = self.hours_for_sensor(sid, start, now_utc)
                if series_h:
                    acc.extend(self._normalize_hourly(series_h, sid))
                else:
                    series_m = self.measurements_for_sensor(sid, start, now_utc)
                    acc.extend(self._normalize_hourly(series_m, sid))
                time.sleep(0.25)
            return acc

        rows = pull_for(candidates)

        if not rows and bbox:
            raw_bbox = self.measurements_bbox(bbox, start, now_utc)
            if raw_bbox:
                rows = self._normalize_hourly(raw_bbox, sensor_id=0)

        if not rows:
            raise RuntimeError(f'No v3 hourly data returned for {city!r} in last {hours}h.')
        return rows

class OpenMeteoAQ:
    BASE = 'https://air-quality-api.open-meteo.com/v1/air-quality'
    CITY_LL = {'berlin': (52.52,13.405),'hamburg': (53.55,9.993),'munich': (48.137,11.575)}
    def __init__(self, timeout:int=30) -> None:
        self.timeout = timeout
        self.sess = requests.Session()
    def fetch_pm25_city(self, *, city:str, hours:int) -> List[Dict[str,Any]]:
        latlon = self.CITY_LL.get(city.lower())
        if not latlon: raise SystemExit(f'OpenMeteo fallback needs lat/lon for {city!r}.')
        lat, lon = latlon
        past_days = max(1, math.ceil(hours/24))
        resp = self.sess.get(self.BASE, params={
            'latitude': lat, 'longitude': lon, 'hourly': 'pm2_5', 'past_days': past_days, 'timezone': 'UTC'
        }, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        times = data.get('hourly',{}).get('time') or []
        vals  = data.get('hourly',{}).get('pm2_5') or []
        rows = []
        for t,v in zip(times, vals):
            if v is not None: rows.append({'datetime': t, 'value': v, 'unit': 'µg/m³'})
        return rows[-hours:] if hours and rows else rows

def fetch_pm25_city(city:str, hours:int, prefer:str='auto') -> List[Dict[str,Any]]:
    iso_hint = {'berlin':'DE','hamburg':'DE','munich':'DE'}.get(city.lower())
    if prefer in ('openaq','auto'):
        try:
            return OpenAQv3().fetch_pm25_city(city=city, hours=hours, iso_hint=iso_hint)
        except PermissionError:
            if prefer == 'openaq': raise
        except Exception:
            if prefer == 'openaq': raise
    return OpenMeteoAQ().fetch_pm25_city(city=city, hours=hours)
