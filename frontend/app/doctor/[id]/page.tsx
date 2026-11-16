import { PatientDetail } from "./PatientDetail";

export const metadata = { title: "Patient record · HealthLens" };

export default async function DoctorPatientPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  return <PatientDetail patientId={id} />;
}
