import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts';

function buildSeries(timeAxis, ...valueSeries) {
  if (!timeAxis || !timeAxis.length) return [];
  return timeAxis.map((t, i) => {
    const pt = { time: parseFloat(Number(t).toFixed(2)) };
    valueSeries.forEach(({ key, data }) => {
      pt[key] = data && data[i] != null ? parseFloat(Number(data[i]).toFixed(4)) : 0;
    });
    return pt;
  });
}

function MiniGraph({ title, subtitle, data, dataKeys, unit = '', color = '#38bdf8' }) {
  if (!data || !data.length) return null;
  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.65)',
      border: '1px solid rgba(148, 163, 184, 0.15)',
      borderRadius: '12px',
      padding: '0.85rem 1rem',
      marginBottom: '1rem'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.25rem' }}>
        <strong style={{ fontSize: '0.85rem', color: '#f1f5f9' }}>{title}</strong>
        <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{subtitle}</span>
      </div>

      <div style={{ width: '100%', height: 130 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 10, left: -22, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 10, fill: '#64748b' }}
              tickFormatter={(v) => `${v}s`}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#64748b' }}
              unit={unit}
              width={35}
            />
            <Tooltip
              contentStyle={{ background: '#0b1322', border: '1px solid #1e293b', borderRadius: 8, fontSize: 11 }}
              labelFormatter={(v) => `Time: ${v}s`}
            />
            {dataKeys.map(({ key, label, stroke }) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={label || key}
                stroke={stroke || color}
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
            {dataKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function SpectrogramPanel({ result }) {
  const features = result?.feature_extraction || {};
  const isCompleted = features.status === 'completed';

  const time = features.chart_time_axis;
  const rms = features.chart_rms;
  const zcr = features.chart_zcr;
  const pitch = features.chart_pitch;
  const mfcc = features.chart_mfcc || {};

  const rmsData = buildSeries(time, { key: 'rms', data: rms });
  const zcrData = buildSeries(time, { key: 'zcr', data: zcr });
  const pitchData = buildSeries(time, { key: 'pitch', data: pitch });
  const mfccData = buildSeries(
    time,
    { key: 'mfcc_1', data: mfcc.mfcc_1 },
    { key: 'mfcc_2', data: mfcc.mfcc_2 },
    { key: 'mfcc_3', data: mfcc.mfcc_3 }
  );

  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Audio Features & Signal Graphs</span>
        <span className={`pill ${isCompleted ? 'pass' : 'incomplete'}`}>
          {features.status || 'Waiting'}
        </span>
      </div>

      {!result ? (
        <p style={{ color: '#64748b', fontSize: '0.85rem' }}>
          Record and verify voice to generate audio feature graphs.
        </p>
      ) : !isCompleted ? (
        <p style={{ color: '#fcd34d', fontSize: '0.85rem' }}>
          {features.reason || 'Audio features not available.'}
        </p>
      ) : (
        <>
          {/* Quick stats strip */}
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '8px',
            marginBottom: '1rem',
            fontSize: '0.75rem',
            color: '#94a3b8'
          }}>
            <span className="pill" style={{ background: 'rgba(30, 41, 59, 0.8)' }}>
              Frames: {features.frame_counts?.rms_energy || 0}
            </span>
            <span className="pill" style={{ background: 'rgba(30, 41, 59, 0.8)' }}>
              Sample Rate: {features.sample_rate || 16000} Hz
            </span>
            <span className="pill" style={{ background: 'rgba(30, 41, 59, 0.8)' }}>
              6 Feature Sets Extracted
            </span>
          </div>

          {/* Graph 1: RMS Energy (Loudness profile) */}
          <MiniGraph
            title="1. RMS Energy (Voice Volume / Dynamics)"
            subtitle="Natural speech rises & falls"
            data={rmsData}
            dataKeys={[{ key: 'rms', label: 'Loudness' }]}
            color="#4ade80"
          />

          {/* Graph 2: Pitch / F0 in Hz */}
          <MiniGraph
            title="2. Pitch / Intonation Contour (F0)"
            subtitle="Human inflection vs flat robotic tone"
            data={pitchData}
            dataKeys={[{ key: 'pitch', label: 'Pitch (Hz)' }]}
            unit="Hz"
            color="#f472b6"
          />

          {/* Graph 3: Zero-Crossing Rate */}
          <MiniGraph
            title="3. Zero-Crossing Rate (ZCR)"
            subtitle="Distinguishes unvoiced consonants from vowels"
            data={zcrData}
            dataKeys={[{ key: 'zcr', label: 'ZCR' }]}
            color="#60a5fa"
          />

          {/* Graph 4: MFCC Resonances */}
          <MiniGraph
            title="4. MFCCs (Vocal Tract Resonances)"
            subtitle="Spectral envelope used by Deepfake detector"
            data={mfccData}
            dataKeys={[
              { key: 'mfcc_1', label: 'MFCC 1', stroke: '#facc15' },
              { key: 'mfcc_2', label: 'MFCC 2', stroke: '#fb923c' },
              { key: 'mfcc_3', label: 'MFCC 3', stroke: '#a78bfa' }
            ]}
          />

          <div style={{
            fontSize: '0.72rem',
            color: '#64748b',
            lineHeight: '1.4',
            background: 'rgba(15, 23, 42, 0.5)',
            padding: '0.5rem 0.75rem',
            borderRadius: '8px'
          }}>
            💡 <strong>How to explain:</strong> Human speech exhibits natural inflection in pitch (Graph 2) and varied loudness dynamics (Graph 1), whereas cloned or synthetic voices often produce abnormally static spectral patterns or robotic discontinuities in MFCCs (Graph 4).
          </div>
        </>
      )}
    </section>
  );
}
