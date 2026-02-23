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
      {/* Hero Section */}
      <section className="hero">
        <h1 className="title">SkinProAI</h1>
        <p className="tagline">
          Multimodal dermatological analysis powered by MedGemma
          and intelligent tool orchestration.
        </p>

        <button className="cta-btn" onClick={() => setShowNewPatient(true)}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Patient
        </button>
      </section>

      {/* Existing patients */}
      {patients.length > 0 && (
        <section className="patients-section">
          <p className="section-label">Recent Patients</p>
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
          </div>
        </section>
      )}

      {/* How It Works */}
      <section className="how-section">
        <p className="section-label">How It Works</p>
        <div className="steps-row">
          <div className="step-card">
            <div className="step-num">1</div>
            <h3>Upload</h3>
            <p>Capture or upload a dermatoscopic or clinical image of the lesion.</p>
          </div>
          <div className="step-card">
            <div className="step-num">2</div>
            <h3>Analyze</h3>
            <p>MedGemma examines the image and coordinates specialist tools for deeper insight.</p>
          </div>
          <div className="step-card">
            <div className="step-num">3</div>
            <h3>Track</h3>
            <p>Monitor lesions over time with side-by-side comparison and change detection.</p>
          </div>
        </div>
      </section>

      {/* About */}
      <section className="about-section">
        <div className="about-card">
          <h3>About SkinProAI</h3>
          <p>
            Built for the <strong>Kaggle MedGemma Multimodal Medical AI Competition</strong>,
            SkinProAI explores how a foundation medical vision-language model can be
            augmented with specialised tools to deliver richer clinical insight.
          </p>
          <p>
            At its core sits Google's <strong>MedGemma 4B</strong>, a multimodal model
            fine-tuned for medical image understanding. Rather than relying on the model
            alone, SkinProAI connects it to a suite of external tools via
            the <strong>Model Context Protocol (MCP)</strong> &mdash; including MONET
            feature extraction, ConvNeXt classification, Grad-CAM attention maps, and
            clinical guideline retrieval &mdash; letting the model reason across multiple
            sources before presenting a synthesised assessment.
          </p>
          <div className="tech-pills">
            <span className="pill">MedGemma 4B</span>
            <span className="pill">MCP Tools</span>
            <span className="pill">MONET</span>
            <span className="pill">ConvNeXt</span>
            <span className="pill">Grad-CAM</span>
            <span className="pill">RAG Guidelines</span>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <footer className="disclaimer">
        <p>
          <strong>Research prototype only.</strong> SkinProAI is an educational project and
          competition entry. It is not a medical device and must not be used for clinical
          decision-making. Always consult a qualified healthcare professional for diagnosis
          and treatment.
        </p>
      </footer>

      {/* New Patient Modal */}
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
