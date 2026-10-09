export default function TranscriptPanel({ result }) {
  const phrase = result?.phrase_verification || {};
  return (
    <section className="card">
      <div className="section-header">
        <span className="eyebrow">Phrase verification</span>
      </div>
      <ul>
        <li>Expected phrase: {phrase.expected_phrase || 'Not available'}</li>
        <li>Recognized phrase: {phrase.recognized_phrase || 'Not available'}</li>
        <li>Exact normalized match: {phrase.exact_match !== undefined ? String(phrase.exact_match) : 'N/A'}</li>
        <li>Similarity: {phrase.similarity ?? 'N/A'}</li>
        <li>Missing words: {(phrase.missing_words || []).join(', ') || 'None'}</li>
        <li>Extra words: {(phrase.extra_words || []).join(', ') || 'None'}</li>
      </ul>
    </section>
  );
}
