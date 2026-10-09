import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts';

/* ── helpers ─────────────────────────────────────────────────────── */

function scorePercent(v) {
  if (v == null || isNaN(v)) return null;
  return Math.round(Math.min(1, Math.max(0, v)) * 100);
}

function Gauge({ value, color, size = 100 }) {
  const pct = value ?? 0;
  return (
    <ResponsiveContainer width={size} height={size}>
      <RadialBarChart
        cx="50%" cy="50%"
        innerRadius="65%" outerRadius="100%"
        data={[{ value: pct }]}
        startAngle={90} endAngle={90 - 360 * (pct / 100)}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar dataKey="value" fill={color} background={{ fill: '#1e293b' }} />
      </RadialBarChart>
    </ResponsiveContainer>
  );
}

function ScoreRow({ label, value, color }) {
  const pct = scorePercent(value);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
      <div style={{ width: 64, height: 64, flexShrink: 0 }}>
        <Gauge value={pct} color={color} size={64} />
      </div>
      <div>
        <div style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
        <div style={{ fontSize: '1.5rem', fontWeight: 700, color }}>
          {pct != null ? `${pct}%` : 'N/A'}
        </div>
      </div>
    </div>
  );
}

/* ── main component ───────────────────────────────────────────────── */

export default function VoiceVerdictCard({ result }) {
  if (!result) {
    return (
      <section className="card">
        <div className="section-header">
          <span className="eyebrow">Voice verdict</span>
        </div>
        <p style={{ color: '#475569', fontSize: '0.85rem' }}>Submit a recording to see the verdict.</p>
      </section>
    );
  }

  const spoof     = result.spoof_detection || {};
  const phrase    = result.phrase_verification || {};
  const quality   = result.audio_quality || {};
  const vad       = result.speech_detection || {};

  const genuineScore = spoof.genuine_score;
  const spoofScore   = spoof.spoof_score;
  const predicted    = spoof.predicted_class;   // 'genuine' | 'spoof' | null
  const spoofStatus  = spoof.status;            // 'completed' | 'unavailable' | 'error' | 'incomplete'

  /* ── determine verdict ────────────────────────────────────────── */
  let verdict       = 'UNKNOWN';
  let verdictColor  = '#64748b';
  let verdictIcon   = '❓';
  let verdictLabel  = 'Insufficient data';
  let verdictSub    = 'The anti-spoof model did not return a result.';

  if (spoofStatus === 'completed' && predicted) {
    if (predicted === 'genuine') {
      verdict      = 'HUMAN';
      verdictColor = '#4ade80';
      verdictIcon  = '✅';
      verdictLabel = 'Likely Human Voice';
      verdictSub   = 'The anti-spoof detector classifies this recording as genuine speech.';
    } else {
      verdict      = 'AI / DEEPFAKE';
      verdictColor = '#f87171';
      verdictIcon  = '⚠️';
      verdictLabel = 'Possible AI-Generated / Deepfake';
      verdictSub   = 'The anti-spoof detector classifies this recording as synthetic or spoofed speech.';
    }
  } else if (spoofStatus === 'unavailable') {
    verdict      = 'MODEL UNAVAILABLE';
    verdictColor = '#f59e0b';
    verdictIcon  = '🔶';
    verdictLabel = 'Anti-Spoof Model Not Loaded';
    verdictSub   = spoof.reason || 'Configure a spoof detection model to get a verdict.';
  } else if (spoofStatus === 'error') {
    verdict      = 'ERROR';
    verdictColor = '#f87171';
    verdictIcon  = '🔴';
    verdictLabel = 'Detection Error';
    verdictSub   = spoof.reason || 'An error occurred during anti-spoof inference.';
  }

  /* ── build checklist ─────────────────────────────────────────── */
  const checks = [
    {
      label: 'Audio quality',
      pass: quality.status === 'pass' || quality.status === 'suspicious',
      note: quality.status === 'suspicious' ? 'Clipped — analysis continued' : quality.reason,
    },
    {
      label: 'Speech detected (VAD)',
      pass: vad.speech_detected === true,
      note: vad.speech_detected ? `${vad.speech_duration_seconds?.toFixed(2)}s of speech found` : 'No speech detected',
    },
    {
      label: 'Phrase matched',
      pass: phrase.exact_match === true,
      note: phrase.exact_match === true
        ? `"${phrase.recognized_phrase}"`
        : phrase.recognized_phrase
          ? `Got: "${phrase.recognized_phrase}"`
          : 'No transcription available',
    },
    {
      label: 'Anti-spoof check',
      pass: predicted === 'genuine',
      na:   spoofStatus !== 'completed',
      note: spoofStatus !== 'completed'
        ? verdictSub
        : predicted === 'genuine'
          ? `Genuine score ${scorePercent(genuineScore)}%`
          : `Spoof score ${scorePercent(spoofScore)}%`,
    },
  ];

  return (
    <section className="card" style={{ border: `1px solid ${verdictColor}33` }}>
      {/* ── big verdict banner ─────────────────────────────────── */}
      <div style={{
        background: `${verdictColor}15`,
        borderRadius: 8,
        padding: '1rem 1.25rem',
        marginBottom: '1.25rem',
        display: 'flex',
        alignItems: 'center',
        gap: '1rem',
      }}>
        <span style={{ fontSize: '2rem' }}>{verdictIcon}</span>
        <div>
          <div style={{ fontSize: '0.65rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            FUSION Voice Verdict
          </div>
          <div style={{ fontSize: '1.25rem', fontWeight: 800, color: verdictColor, lineHeight: 1.2 }}>
            {verdictLabel}
          </div>
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            {verdictSub}
          </div>
        </div>
      </div>

      {/* ── score gauges ─────────────────────────────────────────── */}
      {spoofStatus === 'completed' && (
        <div style={{ display: 'flex', gap: '1.5rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
          <ScoreRow label="Genuine"  value={genuineScore} color="#4ade80" />
          <ScoreRow label="Spoof"    value={spoofScore}   color="#f87171" />
        </div>
      )}

      {/* ── checklist ────────────────────────────────────────────── */}
      <div style={{ marginBottom: '0.75rem' }}>
        <p style={{ fontSize: '0.72rem', color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '0.5rem' }}>
          Check breakdown
        </p>
        {checks.map(({ label, pass, na, note }) => (
          <div key={label} style={{
            display: 'flex', alignItems: 'flex-start', gap: '0.6rem',
            padding: '0.4rem 0', borderBottom: '1px solid #1e293b',
          }}>
            <span style={{ fontSize: '1rem', flexShrink: 0, marginTop: 1 }}>
              {na ? '⬜' : pass ? '✅' : '❌'}
            </span>
            <div>
              <div style={{ fontSize: '0.82rem', color: na ? '#64748b' : pass ? '#e2e8f0' : '#fca5a5', fontWeight: 500 }}>
                {label}
              </div>
              {note && (
                <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.1rem' }}>{note}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ── model info & disclaimer ──────────────────────────────── */}
      {spoof.model_name && (
        <p style={{ fontSize: '0.72rem', color: '#475569', marginTop: '0.5rem' }}>
          Model: <strong style={{ color: '#64748b' }}>{spoof.model_name}</strong>
          {spoof.model_version ? ` v${spoof.model_version}` : ''}
          {spoof.processing_duration_ms ? ` · ${spoof.processing_duration_ms} ms` : ''}
        </p>
      )}
      <p style={{ fontSize: '0.68rem', color: '#334155', marginTop: '0.5rem', lineHeight: 1.5 }}>
        ⚠ Scores are not calibrated probabilities. A genuine verdict does not guarantee the speaker is who they claim to be.
        Anti-spoof analysis is one signal — not a final security decision.
      </p>
    </section>
  );
}
