import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { Patient } from '../types';
import './PatientsPage.css';

export function PatientsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewPatient, setShowNewPatient] = useState(false);
  const [newPatientName, setNewPatientName] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    loadPatients();
  }, []);

  const loadPatients = () => {
    api.listPatients()
      .then(res => setPatients(res.patients))
      .finally(() => setLoading(false));
  };

  const handleCreatePatient = async () => {
    if (!newPatientName.trim()) return;

    const { patient } = await api.createPatient(newPatientName.trim());
    setPatients(prev => [...prev, patient]);
    setNewPatientName('');
    setShowNewPatient(false);
    navigate(`/chat/${patient.id}`);
  };

  const handleDeletePatient = async (e: React.MouseEvent, patientId: string) => {
    e.stopPropagation();
    if (!confirm('Delete this patient and all their data?')) return;

    await api.deletePatient(patientId);
    setPatients(prev => prev.filter(p => p.id !== patientId));
  };

  if (loading) {
    return (
      <div className="patients-page">
        <div className="loading">Loading...</div>
      </div>
    );
  }

  return (
    <div className="patients-page">
      <h1 className="title">SkinProAI</h1>
      <p className="subtitle">Dermatological AI Assistant</p>

      <div className="patients-grid">
        {patients.map(patient => (
          <div
            key={patient.id}
            className="patient-card"
            onClick={() => navigate(`/chat/${patient.id}`)}
          >
            <div className="patient-icon">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
              </svg>
            </div>
            <span className="patient-name">{patient.name}</span>
            <button
              className="delete-btn"
              onClick={(e) => handleDeletePatient(e, patient.id)}
              title="Delete patient"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
              </svg>
            </button>
          </div>
        ))}

        <div
          className="patient-card new-patient"
          onClick={() => setShowNewPatient(true)}
        >
          <div className="add-icon">+</div>
          <span className="patient-name">New Patient</span>
        </div>
      </div>

      {showNewPatient && (
        <div className="modal-overlay" onClick={() => setShowNewPatient(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2>New Patient</h2>
            <input
              type="text"
              placeholder="Patient name..."
              value={newPatientName}
              onChange={e => setNewPatientName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCreatePatient()}
              autoFocus
            />
            <div className="modal-buttons">
              <button className="cancel-btn" onClick={() => setShowNewPatient(false)}>
                Cancel
              </button>
              <button
                className="create-btn"
                onClick={handleCreatePatient}
                disabled={!newPatientName.trim()}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
