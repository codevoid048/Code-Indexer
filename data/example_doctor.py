# Example Python file to demonstrate the code indexer
from typing import List, Dict, Optional
import datetime


class Doctor:
    """A doctor class to demonstrate code indexing."""
    
    def __init__(self, name: str, specialization: str, address: str):
        self.name = name
        self.specialization = specialization
        self.address = address
        self.patients: List['Patient'] = []
    
    def diagnose(self, symptoms: List[str]) -> str:
        """Diagnose a patient based on symptoms."""
        if not symptoms:
            return "No symptoms provided"
        
        # Simple diagnostic logic
        if "fever" in symptoms and "cough" in symptoms:
            return "Possible flu"
        elif "headache" in symptoms:
            return "Possible tension headache"
        else:
            return "Further examination needed"
    
    def prescribe_medication(self, diagnosis: str, patient: 'Patient') -> Dict[str, str]:
        """Prescribe medication based on diagnosis."""
        medications = {
            "Possible flu": "Rest and fluids",
            "Possible tension headache": "Pain reliever",
            "default": "Consult specialist"
        }
        
        prescription = medications.get(diagnosis, medications["default"])
        
        return {
            "patient": patient.name,
            "diagnosis": diagnosis,
            "medication": prescription,
            "date": datetime.datetime.now().isoformat()
        }
    
    def get_patient_count(self) -> int:
        """Get the number of patients."""
        return len(self.patients)
    
    async def schedule_appointment(self, patient: 'Patient', date: datetime.date) -> bool:
        """Schedule an appointment with a patient."""
        # Async method example
        try:
            # Simulate scheduling logic
            if date > datetime.date.today():
                return True
            return False
        except Exception as e:
            print(f"Error scheduling appointment: {e}")
            return False


class Patient:
    """A patient class."""
    
    def __init__(self, name: str, age: int, medical_history: List[str] = None):
        self.name = name
        self.age = age
        self.medical_history = medical_history or []
        self.appointments: List[datetime.date] = []
    
    def add_medical_record(self, record: str) -> None:
        """Add a medical record."""
        self.medical_history.append(record)
    
    def get_age_category(self) -> str:
        """Get age category."""
        if self.age < 18:
            return "minor"
        elif self.age < 65:
            return "adult" 
        else:
            return "senior"


# Module-level functions
def create_doctor(name: str, specialization: str, address: str) -> Doctor:
    """Factory function to create a doctor."""
    return Doctor(name, specialization, address)


def find_doctor_by_specialization(doctors: List[Doctor], specialization: str) -> Optional[Doctor]:
    """Find a doctor by specialization."""
    for doctor in doctors:
        if doctor.specialization.lower() == specialization.lower():
            return doctor
    return None


# Constants
MAX_PATIENTS_PER_DOCTOR = 100
SPECIALIZATIONS = [
    "General Practice",
    "Cardiology", 
    "Neurology",
    "Orthopedics",
    "Pediatrics"
]

# Module-level variable
hospital_name = "City General Hospital"