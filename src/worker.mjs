import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";

await tf.setBackend("cpu");

const W1 = tf.tensor2d([
  0.9,-0.2, 0.1, 0.4, 0.3, 0.2,-0.1, 0.6,
 -0.1, 0.7,-0.2, 0.3, 0.4,-0.1, 0.5,-0.3,
 -0.3,-0.1, 0.8,-0.4, 0.2, 0.6, 0.3, 0.1,
  0.2, 0.1,-0.3, 0.7,-0.2, 0.5, 0.2, 0.4
], [4,8]);
const b1 = tf.tensor1d([0.1, 0.0, 0.05, 0.08, -0.02, 0.03, 0.06, 0.01]);
const W2 = tf.tensor2d([
  0.4, 0.2,-0.1,
  0.6,-0.2,-0.1,
 -0.3, 0.5, 0.2,
  0.7, 0.1,-0.2,
 -0.1, 0.6, 0.4,
  0.2, 0.4, 0.3,
  0.5,-0.3, 0.2,
  0.3, 0.2, 0.5
], [8,3]);
const b2 = tf.tensor1d([0.05, 0.05, 0.1]);

const dense1 = tf.layers.dense({ units: 8, activation: "relu", inputShape: [4] });
const dense2 = tf.layers.dense({ units: 3, activation: "sigmoid" });
const model  = tf.sequential({ layers: [dense1, dense2] });
dense1.setWeights([W1, b1]);
dense2.setWeights([W2, b2]);

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
function wmoToBucket(code){
  if (code === 0) return 0;
  if ([1,2,3].includes(code)) return 1;
  if ([45,48].includes(code)) return 2;
  if ((code>=51 && code<=67) || (code>=80 && code<=82)) return 3;
  if ((code>=71 && code<=77) || [85,86].includes(code)) return 4;
  if ([95,96,99].includes(code)) return 5;
  return 1;
}
function featuresFromWeather(wx){
  const tempC = wx.temperature;
  const tempNorm = (clamp(tempC, -10, 40) + 10) / 50;
  const bucket = wmoToBucket(wx.weathercode);
  const bucketNorm = bucket / 5;
  const isDay = wx.is_day ? 1 : 0;
  const hour = Number((wx.time || "00:00").split("T")[1]?.slice(0,2) || 0);
  const hourNorm = hour / 23;
  return { tempC, tempNorm, bucket, bucketNorm, isDay, hour, hourNorm };
}
function inferMood(f){
  const x = tf.tensor2d([[f.tempNorm, f.hourNorm, f.bucketNorm, f.isDay]]);
  const [energy, valence, acousticness] = model.predict(x).dataSync();
  x.dispose();
  return { energy, valence, acousticness };
}
function pickSeeds(energy, valence, bucket, tempC){
  const hot = tempC >= 28, cold = tempC <= 10, rain = bucket === 3, storm = bucket === 5;
  if (storm) return ["metal","rock","punk"];
  if (rain && valence < 0.5) return ["chill","ambient","acoustic"];
  if (cold && valence < 0.5) return ["acoustic","folk","jazz"];
  if (hot && energy > 0.6) return ["latin","dance","reggaeton"];
  if (energy > 0.7 && valence > 0.6) return ["pop","dance","edm"];
  if (energy < 0.4 && valence > 0.6) return ["bossanova","jazz","acoustic"];
  if (energy < 0.4 && valence < 0.4) return ["ambient","classical","piano"];
  return ["indie","alt-rock","pop"];
}
const normalizeGenre = s => s.toLowerCase().replace(/\s+/g,'-').replace(/[^a-z0-9-]/g,'');

function quantize(n, step = 0.05){ return Math.round(n/step)*step; }
const WX_TTL_MS = 5 * 60 * 1000;
const wxMemCache = new Map();
function wxCacheKey(lat, lon){
  return `${quantize(lat,0.05).toFixed(2)},${quantize(lon,0.05).toFixed(2)}`;
}
function wxCacheGet(k){ const e = wxMemCache.get(k); return e && (Date.now()-e.ts) < WX_TTL_MS ? e.data : null; }
function wxCachePut(k, d){ wxMemCache.set(k, { ts: Date.now(), data: d }); }

async function fetchWithBackoff(url, init, attempts = 3){
  for (let i=0;i<attempts;i++){
    const r = await fetch(url, init);
    if (r.status !== 429 || i === attempts-1) return r;
    const wait = (2**i)*500 + Math.random()*250;
    await new Promise(res => setTimeout(res, wait));
  }
}

async function fetchWeatherOpenMeteo(env, lat, lon){
  const url = new URL(env.OPEN_METEO_URL);
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("current_weather", "true");
  url.searchParams.set("timezone", "auto");
  const r = await fetchWithBackoff(url.toString());
  if (!r.ok) throw new Error(`Open-Meteo: ${r.status} ${await r.text()}`);
  const j = await r.json();
  return j.current_weather;
}
function symbolToWmoLike(symbol){
  const s = String(symbol || "").toLowerCase();
  if (s.includes("thunder")) return 95;
  if (s.includes("snow") || s.includes("sleet")) return 75;
  if (s.includes("rain")) return 60;
  if (s.includes("fog")) return 45;
  if (s.includes("clearsky")) return 0;
  return 3;
}
async function fetchWeatherMetNo(env, lat, lon){
  const url = `https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=${lat}&lon=${lon}`;
  const ua = env.METNO_UA || "no24.app/1.0 (contact: example@example.com)";
  const r = await fetch(url, { headers: { "User-Agent": ua } });
  if (!r.ok) throw new Error(`METNO: ${r.status} ${await r.text()}`);
  const j = await r.json();
  const ts = j.properties?.timeseries?.[0];
  if (!ts) throw new Error("METNO: empty timeseries");
  const temp = ts.data?.instant?.details?.air_temperature;
  const symbol = ts.data?.next_1_hours?.summary?.symbol_code || ts.data?.next_6_hours?.summary?.symbol_code || "";
  const weathercode = symbolToWmoLike(symbol);
  const iso = ts.time || "";
  const hour = Number(iso.slice(11,13) || 0);
  const is_day = hour >= 7 && hour < 19 ? 1 : 0;
  return { temperature: temp, weathercode, is_day, time: iso };
}
async function fetchWeather(env, lat, lon){
  const key = wxCacheKey(lat, lon);
  const c = wxCacheGet(key);
  if (c) return c;
  try {
    const cw = await fetchWeatherOpenMeteo(env, lat, lon);
    wxCachePut(key, cw);
    return cw;
  } catch(_) {}
  const cw2 = await fetchWeatherMetNo(env, lat, lon);
  wxCachePut(key, cw2);
  return cw2;
}

async function getSpotifyToken(env){
  if (!env.SPOTIFY_CLIENT_ID || !env.SPOTIFY_CLIENT_SECRET) {
    throw new Error("SPOTIFY_SECRETS_MISSING: set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET");
  }
  const creds = btoa(`${env.SPOTIFY_CLIENT_ID}:${env.SPOTIFY_CLIENT_SECRET}`);
  const res = await fetch("https://accounts.spotify.com/api/token", {
    method: "POST",
    headers: {
      "Authorization": `Basic ${creds}`,
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: "grant_type=client_credentials"
  });
  const txt = await res.text();
  if (!res.ok) throw new Error(`SPOTIFY_TOKEN_FAILED ${res.status}: ${txt}`);
  const data = JSON.parse(txt);
  globalThis.__spToken = { token: data.access_token, exp: Date.now() + (data.expires_in - 60) * 1000 };
  return data.access_token;
}
async function getAvailableGenres(env){
  if (globalThis.__genres && (Date.now() - globalThis.__genres.ts) < 3600_000) {
    return globalThis.__genres.list;
  }
  let list = null;
  try {
    const token = await getSpotifyToken(env);
    const r = await fetch("https://api.spotify.com/v1/recommendations/available-genre-seeds", {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (r.ok) {
      const j = await r.json();
      list = new Set(j.genres || []);
    }
  } catch {}
  if (!list) {
    list = new Set(["pop","rock","dance","edm","latin","reggaeton","indie","alt-rock","jazz","ambient","classical","acoustic","hip-hop","r-n-b","metal","punk","bossanova","chill","folk","piano"]);
  }
  globalThis.__genres = { list, ts: Date.now() };
  return list;
}

function seedTerms(seeds = []) {
  const map = {
    "alt-rock": "alternative rock",
    "bossanova": "bossa nova",
    "r-n-b": "r&b",
    "hip-hop": "hip hop",
    "lo-fi": "lofi"
  };
  return Array.from(new Set(
    (seeds || []).map(s => (map[s] || s).replace(/-/g, " ")).filter(Boolean)
  ));
}

async function searchTrackByMood(env, mood, seeds, market = "US"){
  const token = await getSpotifyToken(env);
  const terms = seedTerms(seeds);
  const yearNow = new Date().getUTCFullYear();

  async function doSearch(q, limit = 30) {
    const params = new URLSearchParams({ q, type: "track", market, limit: String(limit) });
    const r = await fetch(`https://api.spotify.com/v1/search?${params}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    const txt = await r.text();
    if (!r.ok) throw new Error(`SPOTIFY_SEARCH_FAILED ${r.status}: ${txt}`);
    const j = JSON.parse(txt);
    return j.tracks?.items || [];
  }

  let tracks = [];
  if (terms.length) {
    const q1 = `(${terms.map(t => `"${t}"`).join(" OR ")}) year:${yearNow-7}-${yearNow}`;
    tracks = await doSearch(q1);
  }
  if (!tracks.length && terms.length) {
    const q2 = `(${terms.map(t => `"${t}"`).join(" OR ")})`;
    tracks = await doSearch(q2);
  }
  if (!tracks.length) {
    const q3 = `year:${yearNow-7}-${yearNow}`;
    tracks = await doSearch(q3);
  }
  if (!tracks.length) {
    const q4 = `pop`;
    tracks = await doSearch(q4);
  }
  if (!tracks.length) return {
    id: "4uLU6hMCjMI75M1A2tKUQC",
    name: "Never Gonna Give You Up",
    artists: "Rick Astley",
    url: "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
    preview_url: null
  };

  try {
    const ids = tracks.slice(0, 50).map(t => t.id).join(",");
    const rf = await fetch(`https://api.spotify.com/v1/audio-features?ids=${ids}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (rf.ok) {
      const jf = await rf.json();
      const feats = jf.audio_features || [];
      let best = null, bestScore = Infinity;
      for (let i = 0; i < feats.length; i++) {
        const f = feats[i]; if (!f) continue;
        const score = Math.hypot(
          (f.energy - mood.energy),
          (f.valence - mood.valence),
          ((f.danceability ?? 0.5) - (mood.danceability ?? 0.5))
        );
        if (score < bestScore) { bestScore = score; best = tracks[i]; }
      }
      if (best) return {
        id: best.id,
        name: best.name,
        artists: (best.artists || []).map(a => a.name).join(", "),
        url: `https://open.spotify.com/track/${best.id}`,
        preview_url: best.preview_url
      };
    }
  } catch {}

  const t = tracks[0];
  return {
    id: t.id,
    name: t.name,
    artists: (t.artists || []).map(a => a.name).join(", "),
    url: `https://open.spotify.com/track/${t.id}`,
    preview_url: t.preview_url
  };
}

async function recommendTrack(env, mood, seeds, market = "US"){
  try {
    const avail = await getAvailableGenres(env);
    const clean = Array.from(new Set((seeds || []).map(normalizeGenre).filter(g => avail.has(g))));
    const seedGenres = (clean.length ? clean : ["pop","rock","dance"]).slice(0,5).join(",");
    const token = await getSpotifyToken(env);
    const params = new URLSearchParams({
      limit: "1",
      seed_genres: seedGenres,
      target_energy: mood.energy.toFixed(2),
      target_valence: mood.valence.toFixed(2),
      min_popularity: "40",
      market
    });
    const url = `https://api.spotify.com/v1/recommendations?${params}`;
    const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    const txt = await r.text();
    if (r.status === 404) throw new Error("RECS_DEPRECATED");
    if (!r.ok) throw new Error(`SPOTIFY_RECS_FAILED ${r.status}: ${txt}`);
    const j = JSON.parse(txt);
    const t = j.tracks?.[0];
    if (t) return {
      id: t.id,
      name: t.name,
      artists: (t.artists || []).map(a => a.name).join(", "),
      url: `https://open.spotify.com/track/${t.id}`,
      preview_url: t.preview_url
    };
  } catch {}
  return await searchTrackByMood(env, mood, seeds, market);
}

function json(data, init = {}){
  return new Response(JSON.stringify(data, null, 2), {
    headers: {
      "content-type": "application/json; charset=utf-8",
      "access-control-allow-origin": "*"
    },
    ...init
  });
}

function html(){
  return new Response(`<!doctype html>
<html lang="es">
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>no24</title>
<style>
  :root { color-scheme: dark; }
  html,body{height:100%}
  body{margin:0;background:#000;color:#fff;display:grid;place-items:center;font: clamp(18px, 2.5vw, 28px)/1.3 system-ui,-apple-system,Segoe UI,Roboto,Inter,sans-serif;}
  .center{ text-align:center; }
</style>
<div class="center" id="msg">1. Analizando clima</div>
<script>
(function(){
  var msg = document.getElementById('msg');
  function set(t){ msg.textContent = t; }
  function getPos(){
    return new Promise(function(resolve, reject){
      navigator.geolocation.getCurrentPosition(
        function(p){ resolve({ lat: p.coords.latitude, lon: p.coords.longitude }); },
        function(e){ reject(e); },
        { enableHighAccuracy: true, timeout: 8000 }
      );
    });
  }
  (async function run(){
    try{
      set('1. Analizando clima');
      var pos = await getPos();
      set('2. Consiguiendo una canción');
      var u = new URL(location.href);
      u.pathname = '/api/track';
      u.search = new URLSearchParams({ lat: pos.lat, lon: pos.lon }).toString();
      var r = await fetch(u.toString());
      var j = await r.json();
      if(!r.ok || !j || !j.track || !j.track.url) throw new Error(j && j.error || 'Sin canción');
      for (var n = 3; n >= 1; n--) {
        set('3. Redireccionando en ' + n);
        await new Promise(function(res){ setTimeout(res, 1000); });
      }
      location.replace(j.track.url);
    } catch(e){
      set('No pude obtener ubicación o canción. Revisa permisos y recarga.');
    }
  })();
})();
</script>
</html>`, { headers: { "content-type": "text/html; charset=utf-8" } });
}

export default {
  async fetch(req, env){
    const url = new URL(req.url);
    const { pathname, searchParams } = url;
    const cfCountry = req.cf?.country;
    const market = /^[A-Z]{2}$/.test(cfCountry || '') ? cfCountry : 'US';

    if (pathname === "/") return html();

    if (pathname === "/api/ping"){
      try {
        const lat = Number(searchParams.get("lat") || 0);
        const lon = Number(searchParams.get("lon") || 0);
        const cw = (Number.isFinite(lat) && Number.isFinite(lon)) ? await fetchWeather(env, lat, lon) : { ok: true };
        return json({ ok: true, weather: cw });
      } catch (e) {
        return json({ ok: false, error: String(e?.message || e) }, { status: 500 });
      }
    }

    if (pathname === "/api/track"){
      const debug = searchParams.get("debug") === "1";
      try{
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
          return json({ error: "lat/lon requeridos" }, { status: 400 });
        }
        const cw   = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood = inferMood(feat);
        const seeds = pickSeeds(mood.energy, mood.valence, feat.bucket, feat.tempC);
        const track = await recommendTrack(env, mood, seeds, market);
        if (!track) return json({ error: "Sin recomendaciones", weather: cw, features: mood, seeds }, { status: 502 });
        const buckets = ["despejado","nublado","niebla","lluvia","nieve","tormenta"];
        const summary = buckets[feat.bucket] + " \u2022 " + (cw.is_day ? "día" : "noche");
        const payload = { weather: { ...cw, summary }, features: mood, seeds, track };
        return json(debug ? { debug: true, ...payload } : payload);
      } catch (err){
        return json({ error: String(err?.message || err), stack: debug ? (err?.stack || null) : undefined }, { status: 500 });
      }
    }

    if (pathname === "/go"){
      try{
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        const cw   = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood = inferMood(feat);
        const seeds = pickSeeds(mood.energy, mood.valence, feat.bucket, feat.tempC);
        const track = await recommendTrack(env, mood, seeds, market);
        const loc = track ? track.url : "https://open.spotify.com/";
        return Response.redirect(loc, 302);
      } catch {
        return Response.redirect("https://open.spotify.com/", 302);
      }
    }

    return new Response("Not found", { status: 404 });
  }
};
