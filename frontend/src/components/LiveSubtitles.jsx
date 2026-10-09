import { useMemo } from 'react';

export default function LiveSubtitles({ phrase, liveTranscript, recognizedPhrase, isRecording }) {
  // Compute normalized word matching for follow-along highlights
  const challengeWords = useMemo(() => {
    if (!phrase) return [];
    return phrase.split(/\s+/);
  }, [phrase]);

  const spokenWordsNormalized = useMemo(() => {
    const text = (isRecording ? liveTranscript : recognizedPhrase || liveTranscript) || '';
    return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
  }, [liveTranscript, recognizedPhrase, isRecording]);

  const currentDisplayText = isRecording
    ? liveTranscript
    : recognizedPhrase || liveTranscript;

  return (
    <div className="subtitle-container">
      {/* Subtitle Banner Overlay */}
      <div className={`subtitle-banner ${isRecording ? 'recording' : ''}`}>
        <div className="subtitle-header">
          <span className="subtitle-badge">
            {isRecording ? (
              <>
                <span className="pulse-dot"></span> LIVE SUBTITLES
              </>
            ) : (
              '💬 SUBTITLES'
            )}
          </span>
          <span className="subtitle-hint">
            {isRecording ? 'Speak the words on screen...' : 'Recognized Speech Transcript'}
          </span>
        </div>

        <div className="subtitle-text-box">
          {currentDisplayText ? (
            <p className="subtitle-text">“{currentDisplayText}”</p>
          ) : (
            <p className="subtitle-placeholder">
              {isRecording
                ? 'Listening to microphone... speak now to see live subtitles.'
                : 'Subtitles will appear here when you record or submit audio.'}
            </p>
          )}
        </div>

        {/* Word Follow-Along Grid */}
        {challengeWords.length > 0 && (
          <div className="follow-along-section">
            <span className="follow-along-label">Follow Along Status:</span>
            <div className="word-chips">
              {challengeWords.map((word, idx) => {
                const cleanWord = word.toLowerCase().replace(/[^a-z0-9]/g, '');
                const isSpoken = spokenWordsNormalized.includes(cleanWord);
                return (
                  <span
                    key={`${word}-${idx}`}
                    className={`word-chip ${isSpoken ? 'spoken' : ''}`}
                  >
                    {word}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
