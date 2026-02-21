import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { PatientsPage } from './pages/PatientsPage';
import { ChatPage } from './pages/ChatPage';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PatientsPage />} />
        <Route path="/chat/:patientId" element={<ChatPage />} />
      </Routes>
    </BrowserRouter>
  );
}
