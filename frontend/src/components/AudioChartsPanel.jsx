import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';

function buildSeries(timeAxis, ...valueSeries) {
  if (!timeAxis || !timeAxis.length) return [];
  return timeAxis.map((t, i) => {
    const pt = { time: parseFloat(t.toFixed(3)) };
    valueSeries.forEach(({ key, data }) => {
      pt[key] = data && data[i] != null ? parseFloat(data[i].toFixed(5)) : null;
    });
    return pt;
  });
}

const CHART_COLORS = {
  rms:    '#4ade80',
  zcr:    '#60a5fa',
  pitch:  '#f472b6',
  mfcc_1: '#facc15',
  mfcc_2: '#fb923c',
  mfcc_3: '#a78bfa',
};

function MiniChart({ title, data, dataKeys, unit = '' }) {
  if (!data || !data.length) return null;
  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <p style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.4rem', letterSpacing: '0.05em' }}>
        {title}
      </p>
      <ResponsiveContainer width="100%" height={130}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: '#64748b' }}
            tickFormatter={(v) => `${v}s`}
          />
          <YAxis tick={{ fontSize: 10, fill: '#64748b' }} unit={unit} />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 6, fontSize: 11 }}
            labelFormatter={(v) => `${v}s`}
          />
          {dataKeys.map(({ key, label }) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              name={label || key}
              stroke={CHART_COLORS[key] || '#94a3b8'}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
          {dataKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function AudioChartsPanel({ result }) {
  const features = result?.feature_extraction || {};
  const time  = features.chart_time_axis;
  const rms   = features.chart_rms;
  const zcr   = features.chart_zcr;
  const pitch = features.chart_pitch;
  const mfcc  = features.chart_mfcc || {};

  const rmsData   = buildSeries(time, { key: 'rms',   data: rms });
  const zcrData   = buildSeries(time, { key: 'zcr',   data: zcr });
  const pitchData = buildSeries(time, { key: 'pitch', data: pitch });
  const mfccData  = buildSeries(
    time,
    { key: 'mfcc_1', data: mfcc.mfcc_1 },
    { key: 'mfcc_2', data: mfcc.mfcc_2 },
    { key: 'mfcc_3', data: mfcc.mfcc_3 },
  );

  const hasData = features.status === 'completed' && time && time.length > 0;

  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Audio feature charts</span>
        <span className={`pill ${features.status || 'incomplete'}`}>
          {features.status || 'Waiting…'}
        </span>
      </div>

      {!result && (
        <p style={{ color: '#475569', fontSize: '0.85rem' }}>Submit a recording to see feature charts.</p>
      )}

      {result && !hasData && (
        <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
          {features.reason || 'Feature extraction did not complete.'}
        </p>
      )}

      {hasData && (
        <>
          <MiniChart
            title="RMS Energy (loudness over time)"
            data={rmsData}
            dataKeys={[{ key: 'rms', label: 'RMS' }]}
          />
          <MiniChart
            title="Zero-Crossing Rate (noisiness / fricative content)"
            data={zcrData}
            dataKeys={[{ key: 'zcr', label: 'ZCR' }]}
          />
          <MiniChart
            title="Pitch / Fundamental Frequency"
            data={pitchData}
            dataKeys={[{ key: 'pitch', label: 'F0 (Hz)' }]}
            unit="Hz"
          />
          <MiniChart
            title="MFCC Coefficients 1–3 (spectral envelope shape)"
            data={mfccData}
            dataKeys={[
              { key: 'mfcc_1', label: 'MFCC-1' },
              { key: 'mfcc_2', label: 'MFCC-2' },
              { key: 'mfcc_3', label: 'MFCC-3' },
            ]}
          />
          <p style={{ fontSize: '0.72rem', color: '#475569', marginTop: '0.5rem' }}>
            Frames: RMS {features.frame_counts?.rms_energy} · Pitch {features.frame_counts?.pitch} · Sample rate {features.sample_rate} Hz
          </p>
        </>
      )}
    </section>
  );
}
