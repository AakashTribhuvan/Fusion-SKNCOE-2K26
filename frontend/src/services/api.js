const API_BASE_URL = 'http://localhost:8001';

export async function createSession() {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions`, { method: 'POST' });
  if (!response.ok) {
    throw new Error('Unable to create a verification session');
  }
  return response.json();
}

export async function refreshChallenge(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/challenge`, { method: 'POST' });
  if (!response.ok) {
    throw new Error('Unable to refresh the challenge phrase');
  }
  return response.json();
}

export async function uploadAudio(sessionId, file) {
  const form = new FormData();
  form.append('file', file, file.name || 'recording.webm');
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/audio`, {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Audio upload failed' }));
    throw new Error(payload.detail || 'Audio upload failed');
  }
  return response.json();
}

export async function verifySession(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/verify`, { method: 'POST' });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Verification failed' }));
    throw new Error(payload.detail || 'Verification failed');
  }
  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('Backend health check failed');
  }
  return response.json();
}
