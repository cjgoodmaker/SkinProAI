import { Patient, ChatMessage } from '../types';

const BASE = '/api';

export const api = {
  // Patients
  listPatients: (): Promise<{ patients: Patient[] }> =>
    fetch(`${BASE}/patients`).then(r => r.json()),

  createPatient: (name: string): Promise<{ patient: Patient }> =>
    fetch(`${BASE}/patients`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }).then(r => r.json()),

  getPatient: (patientId: string): Promise<{ patient: Patient }> =>
    fetch(`${BASE}/patients/${patientId}`).then(r => r.json()),

  deletePatient: (patientId: string): Promise<void> =>
    fetch(`${BASE}/patients/${patientId}`, { method: 'DELETE' }).then(() => {}),

  // Chat (patient-level)
  getChatHistory: (patientId: string): Promise<{ messages: ChatMessage[] }> =>
    fetch(`${BASE}/patients/${patientId}/chat`).then(r => r.json()),

  clearChat: (patientId: string): Promise<void> =>
    fetch(`${BASE}/patients/${patientId}/chat`, { method: 'DELETE' }).then(() => {}),
};
