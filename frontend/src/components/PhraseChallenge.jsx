export default function PhraseChallenge({ phrase, onRefresh, loading }) {
  return (
    <section className="card challenge-card">
      <div className="section-header">
        <span className="eyebrow">Verification challenge</span>
        <button className="secondary" onClick={onRefresh} disabled={loading}>
          {loading ? 'Refreshing…' : 'Generate another phrase'}
        </button>
      </div>
      <h2>Read and repeat this phrase</h2>
      <div className="phrase-box">{phrase || 'Generating challenge…'}</div>
    </section>
  );
}
