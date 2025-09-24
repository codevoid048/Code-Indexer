// Example JavaScript file to demonstrate multi-language indexing
import { Patient } from './patient.js';
import axios from 'axios';

/**
 * Hospital management system
 */
export class HospitalManager {
    constructor(hospitalName, location) {
        this.hospitalName = hospitalName;
        this.location = location;
        this.doctors = [];
        this.patients = [];
    }

    /**
     * Add a new doctor to the hospital
     * @param {Object} doctorData - Doctor information
     * @returns {Promise<boolean>} Success status
     */
    async addDoctor(doctorData) {
        try {
            const doctor = {
                id: this.generateId(),
                name: doctorData.name,
                specialization: doctorData.specialization,
                experience: doctorData.experience || 0,
                isAvailable: true
            };

            this.doctors.push(doctor);
            return true;
        } catch (error) {
            console.error('Error adding doctor:', error);
            return false;
        }
    }

    /**
     * Find doctors by specialization
     * @param {string} specialization - Medical specialization
     * @returns {Array} List of matching doctors
     */
    findDoctorsBySpecialization(specialization) {
        return this.doctors.filter(doctor => 
            doctor.specialization.toLowerCase().includes(specialization.toLowerCase())
        );
    }

    /**
     * Schedule an appointment
     * @param {string} patientId - Patient ID
     * @param {string} doctorId - Doctor ID
     * @param {Date} appointmentDate - Appointment date
     * @returns {Object} Appointment details
     */
    scheduleAppointment(patientId, doctorId, appointmentDate) {
        const patient = this.patients.find(p => p.id === patientId);
        const doctor = this.doctors.find(d => d.id === doctorId);

        if (!patient || !doctor) {
            throw new Error('Patient or doctor not found');
        }

        const appointment = {
            id: this.generateId(),
            patientId,
            doctorId,
            date: appointmentDate,
            status: 'scheduled',
            createdAt: new Date()
        };

        return appointment;
    }

    /**
     * Get hospital statistics
     * @returns {Object} Hospital statistics
     */
    getStatistics() {
        return {
            totalDoctors: this.doctors.length,
            totalPatients: this.patients.length,
            availableDoctors: this.doctors.filter(d => d.isAvailable).length,
            specializations: [...new Set(this.doctors.map(d => d.specialization))]
        };
    }

    /**
     * Generate a unique ID
     * @private
     * @returns {string} Unique identifier
     */
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}

/**
 * Utility functions
 */
export const HospitalUtils = {
    /**
     * Validate doctor data
     * @param {Object} data - Doctor data to validate
     * @returns {boolean} Validation result
     */
    validateDoctorData(data) {
        return data.name && 
               data.specialization && 
               typeof data.name === 'string' &&
               typeof data.specialization === 'string';
    },

    /**
     * Format appointment date
     * @param {Date} date - Date to format
     * @returns {string} Formatted date string
     */
    formatAppointmentDate(date) {
        return date.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    },

    /**
     * Calculate doctor experience level
     * @param {number} years - Years of experience
     * @returns {string} Experience level
     */
    getExperienceLevel(years) {
        if (years < 2) return 'junior';
        if (years < 10) return 'experienced';
        return 'senior';
    }
};

// Constants
export const HOSPITAL_DEPARTMENTS = [
    'Emergency',
    'Cardiology',
    'Neurology',
    'Orthopedics',
    'Pediatrics',
    'General Medicine'
];

export const APPOINTMENT_STATUS = {
    SCHEDULED: 'scheduled',
    COMPLETED: 'completed',
    CANCELLED: 'cancelled',
    NO_SHOW: 'no_show'
};

// Default export
export default HospitalManager;