import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-cpu";
const backendReady = tf.setBackend("cpu");
await backendReady;


const W1 = tf.tensor2d([
  0.9, -0.2,  0.1,  0.4,  0.3,  0.2, -0.1,  0.6,
 -0.1,  0.7, -0.2,  0.3,  0.4, -0.1,  0.5, -0.3,
 -0.3, -0.1,  0.8, -0.4,  0.2,  0.6,  0.3,  0.1,
  0.2,  0.1, -0.3,  0.7, -0.2,  0.5,  0.2,  0.4
], [4,8]);
const b1 = tf.tensor1d([0.1, 0.0, 0.05, 0.08, -0.02, 0.03, 0.06, 0.01]);
const W2 = tf.tensor2d([
  0.4,  0.2, -0.1,
  0.6, -0.2, -0.1,
 -0.3,  0.5,  0.2,
  0.7,  0.1, -0.2,
 -0.1,  0.6,  0.4,
  0.2,  0.4,  0.3,
  0.5, -0.3,  0.2,
  0.3,  0.2,  0.5
], [8,3]);
const b2 = tf.tensor1d([0.05, 0.05, 0.1]);

const dense1 = tf.layers.dense({ units: 8, activation: "relu", inputShape: [4] });
const dense2 = tf.layers.dense({ units: 3, activation: "sigmoid" });

const model = tf.sequential({ layers: [dense1, dense2] });
await dense1.setWeights([W1, b1]);
await dense2.setWeights([W2, b2]);

// ————————————————————————————————————————————————————————————————
// Utilidades
// ————————————————————————————————————————————————————————————————

function clamp(v, min, max){ return Math.min(max, Math.max(min, v)); }

function wmoToBucket(code){
  if (code === 0) return 0;
  if ([1,2,3].includes(code)) return 1;
  if ([45,48].includes(code)) return 2;
  if ((code>=51 && code<=67) || (code>=80 && code<=82)) return 3;
  if ((code>=71 && code<=77) || [85,86].includes(code)) return 4;
  if ([95,96,99].includes(code)) return 5;
  return 1;
}

function pickSeeds(energy, valence, bucket, tempC){
  const hot = tempC >= 28;
  const cold = tempC <= 10;
  const rain = bucket === 3;
  const storm = bucket === 5;

  if (storm) return ["metal", "rock", "punk"];
  if (rain && valence < 0.5) return ["lo-fi", "ambient", "chill"];
  if (cold && valence < 0.5) return ["acoustic", "folk", "singer-songwriter"];
  if (hot && energy > 0.6) return ["latin", "dance", "reggaeton"];
  if (energy > 0.7 && valence > 0.6) return ["pop", "dance", "edm"];
  if (energy < 0.4 && valence > 0.6) return ["bossa-nova", "jazz", "acoustic"];
  if (energy < 0.4 && valence < 0.4) return ["ambient", "classical", "piano"];
  return ["indie", "alt-rock", "pop"];
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

function inferMood(feat){
  const x = tf.tensor2d([[feat.tempNorm, feat.hourNorm, feat.bucketNorm, feat.isDay]]);
  const [energy, valence, acousticness] = model.predict(x).dataSync();
  x.dispose();
  return { energy, valence, acousticness };
}

async function getSpotifyToken(env){
  if (globalThis.__spToken && Date.now() < globalThis.__spToken.exp) {
    return globalThis.__spToken.token;
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
  if (!res.ok) throw new Error(`Spotify token failed: ${res.status}`);
  const data = await res.json();
  globalThis.__spToken = {
    token: data.access_token,
    exp: Date.now() + (data.expires_in - 60) * 1000
  };
  return data.access_token;
}

async function fetchWeather(env, lat, lon){
  const url = new URL(env.OPEN_METEO_URL);
  url.searchParams.set("latitude", String(lat));
  url.searchParams.set("longitude", String(lon));
  url.searchParams.set("current_weather", "true");
  url.searchParams.set("timezone", "auto");

  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`Open-Meteo: ${r.status}`);
  const j = await r.json();
  return j.current_weather;
}

async function recommendTrack(env, mood, seeds){
  const token = await getSpotifyToken(env);
  const params = new URLSearchParams({
    limit: "1",
    seed_genres: seeds.slice(0,5).join(","),
    target_energy: mood.energy.toFixed(2),
    target_valence: mood.valence.toFixed(2),
    min_popularity: "40"
  });
  const r = await fetch(`https://api.spotify.com/v1/recommendations?${params}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!r.ok) throw new Error(`Spotify recs: ${r.status}`);
  const j = await r.json();
  const t = j.tracks?.[0];
  if (!t) return null;
  const artists = (t.artists || []).map(a => a.name).join(", ");
  return {
    id: t.id,
    name: t.name,
    artists,
    url: `https://open.spotify.com/track/${t.id}`,
    preview_url: t.preview_url
  };
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
<title>Spotify Weather Redirector</title>
<style>
  :root { color-scheme: dark; }
  html,body{height:100%}
  body{margin:0; background:#000; color:#fff; display:grid; place-items:center; font: clamp(18px, 2.5vw, 28px)/1.3 system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif;}
  .center{ text-align:center; white-space:pre-line }
</style>
<div class="center" id="msg">1. Analizando clima</div>
<script>
(async () => {
  const $ = s => document.querySelector(s);
  const set = t => { $('#msg').textContent = t; };
  try {
    // Paso 1: Analizando clima (solicitamos ubicación)
    set('1. Analizando clima');
    const pos = await new Promise((res, rej) =>
      navigator.geolocation.getCurrentPosition(
        p => res({ lat: p.coords.latitude, lon: p.coords.longitude }),
        e => rej(e),
        { enableHighAccuracy: true, timeout: 8000 }
      )
    );

    // Paso 2: consiguiendo canción
    set('2. Consiguiendo una canción');
    const u = new URL(location.href);
    u.pathname = '/api/track';
    u.search = new URLSearchParams({ lat: pos.lat, lon: pos.lon }).toString();
    const r = await fetch(u.toString());
    const j = await r.json();
    if (!r.ok || !j?.track?.url) throw new Error(j.error || 'Sin canción');

    // Paso 3: redireccionando 3,2,1 (contador en vivo)
    for (let n = 3; n >= 1; n--) {
      set(`3. Redireccionando en ${n}`);
      await new Promise(r => setTimeout(r, 1000));
    }
    location.replace(j.track.url);
  } catch (e) {
    set('No pude obtener ubicación o canción. Revisa permisos y recarga.');
  }
})();
</script>
</html>`, {
    headers: { "content-type": "text/html; charset=utf-8" }
  });
}

// ————————————————————————————————————————————————————————————————
// Ruteo minimalista
// ————————————————————————————————————————————————————————————————

export default {
  async fetch(req, env, ctx){
    const url = new URL(req.url);
    const { pathname, searchParams } = url;

    if (pathname === "/"){
      return html();
    }

    if (pathname === "/api/track"){
      try {
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
          return json({ error: "lat/lon requeridos" }, { status: 400 });
        }

        const cw = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood = inferMood(feat);
        const seeds = pickSeeds(mood.energy, mood.valence, feat.bucket, feat.tempC);
        const track = await recommendTrack(env, mood, seeds);

        if (!track){
          return json({ error: "Sin recomendaciones" }, { status: 502 });
        }

        const summary = (() => {
          const buckets = ["despejado", "nublado", "niebla", "lluvia", "nieve", "tormenta"];
          return `${buckets[feat.bucket]} • ${cw.is_day ? "día" : "noche"}`;
        })();

        return json({
          weather: {
            temperature: cw.temperature,
            weathercode: cw.weathercode,
            is_day: cw.is_day,
            time: cw.time,
            summary
          },
          features: mood,
          seeds,
          track
        });
      } catch (err) {
        return json({ error: String(err.message || err) }, { status: 500 });
      }
    }

    if (pathname === "/go"){
      // Alternativa: /go?lat=..&lon=..
      try {
        const lat = Number(searchParams.get("lat"));
        const lon = Number(searchParams.get("lon"));
        const cw = await fetchWeather(env, lat, lon);
        const feat = featuresFromWeather(cw);
        const mood = inferMood(feat);
        const seeds = pickSeeds(mood.energy, mood.valence, feat.bucket, feat.tempC);
        const track = await recommendTrack(env, mood, seeds);
        if (!track) return new Response("", { status: 302, headers: { Location: "https://open.spotify.com/" }});
        return Response.redirect(track.url, 302);
      } catch (e){
        return new Response("", { status: 302, headers: { Location: "https://open.spotify.com/" }});
      }
    }

    return new Response("Not found", { status: 404 });
  }
};