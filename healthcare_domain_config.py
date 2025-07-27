#!/usr/bin/env python3
"""
Healthcare Domain Configuration for Enhanced RAG Voice Agent
Shows how to customize the agent behavior for specific domains
"""

class HealthcareDomainConfig:
    """Configuration for healthcare-specific RAG behavior"""
    
    # Domain-specific prompts
    SYSTEM_PROMPTS = {
        "healthcare_assistant": """You are a knowledgeable healthcare information assistant. 
        Use the provided medical knowledge to give accurate, evidence-based responses about health topics.
        
        IMPORTANT GUIDELINES:
        - Always include a disclaimer that this is for educational purposes only
        - Recommend consulting healthcare professionals for medical advice
        - Be clear about when to seek emergency care
        - Use simple, understandable language
        - Focus on evidence-based information
        - Avoid diagnosing or prescribing treatments
        
        When asked about symptoms, medications, or health conditions, provide:
        1. Clear, factual information from your knowledge base
        2. General recommendations for prevention/management
        3. Warning signs that require medical attention
        4. Appropriate disclaimers about professional consultation
        """,
        
        "emergency_triage": """You are helping with health emergency triage.
        
        CRITICAL PRIORITIES:
        - Identify emergency situations requiring immediate 911 calls
        - Provide clear, calm guidance for urgent situations
        - Always err on the side of caution
        - Use the knowledge base to identify warning signs
        
        EMERGENCY SITUATIONS (call 911):
        - Chest pain with heart attack symptoms
        - Difficulty breathing or shortness of breath
        - Severe bleeding that won't stop
        - Loss of consciousness
        - Severe allergic reactions
        - Signs of stroke (face drooping, arm weakness, speech difficulty)
        - Thoughts of suicide or self-harm
        """,
        
        "medication_safety": """You are a medication safety assistant.
        
        SAFETY PRIORITIES:
        - Emphasize following healthcare provider instructions
        - Highlight drug interaction risks
        - Stress importance of proper storage and disposal
        - Never recommend changing prescribed medications
        - Encourage pharmacist consultation for questions
        
        ALWAYS INCLUDE:
        - Reminders to consult healthcare providers
        - Warnings about not sharing prescription medications
        - Importance of completing antibiotic courses
        - Proper disposal methods for unused medications
        """
    }
    
    # Domain-specific response templates
    RESPONSE_TEMPLATES = {
        "symptom_query": """Based on the medical information available:

{context}

**Important Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. If you're experiencing concerning symptoms, please consult with a healthcare professional or call your doctor.

{emergency_note}""",
        
        "medication_query": """Here's important medication information:

{context}

**Medication Safety Reminder:** Always follow your healthcare provider's instructions exactly. Never stop, start, or change medications without consulting your doctor or pharmacist. If you have questions about your medications, contact your pharmacy or healthcare provider.

{safety_warnings}""",
        
        "emergency_response": """⚠️ **EMERGENCY GUIDANCE** ⚠️

{context}

🚨 **If this is a medical emergency, call 911 immediately.** 🚨

Do not wait if you're experiencing:
- Chest pain or heart attack symptoms
- Difficulty breathing
- Severe bleeding
- Loss of consciousness
- Signs of stroke

This information is not a substitute for emergency medical care."""
    }
    
    # Query classification patterns
    QUERY_PATTERNS = {
        "emergency": [
            "emergency", "911", "urgent", "severe pain", "can't breathe", 
            "chest pain", "heart attack", "stroke", "bleeding", "unconscious"
        ],
        "symptoms": [
            "symptoms", "signs", "feeling", "pain", "ache", "fever", 
            "headache", "nausea", "cough", "cold", "flu"
        ],
        "medication": [
            "medication", "medicine", "pills", "prescription", "drug", 
            "dosage", "side effects", "interactions"
        ],
        "chronic_condition": [
            "diabetes", "blood pressure", "hypertension", "heart disease",
            "depression", "anxiety", "cholesterol"
        ],
        "prevention": [
            "prevent", "avoid", "reduce risk", "healthy", "diet", 
            "exercise", "lifestyle"
        ]
    }
    
    # Response modifiers based on query type
    RESPONSE_MODIFIERS = {
        "emergency": {
            "urgency_level": "high",
            "include_911_reminder": True,
            "emphasize_professional_care": True
        },
        "symptoms": {
            "include_when_to_see_doctor": True,
            "include_disclaimer": True,
            "provide_general_care_tips": True
        },
        "medication": {
            "emphasize_prescription_compliance": True,
            "warn_about_interactions": True,
            "include_pharmacist_consultation": True
        },
        "chronic_condition": {
            "focus_on_management": True,
            "include_monitoring_tips": True,
            "emphasize_regular_checkups": True
        },
        "prevention": {
            "provide_lifestyle_recommendations": True,
            "include_screening_reminders": True,
            "focus_on_evidence_based_advice": True
        }
    }

def classify_healthcare_query(query_text):
    """Classify the type of healthcare query"""
    query_lower = query_text.lower()
    
    for category, patterns in HealthcareDomainConfig.QUERY_PATTERNS.items():
        if any(pattern in query_lower for pattern in patterns):
            return category
    
    return "general"

def get_healthcare_system_prompt(query_type="general"):
    """Get appropriate system prompt based on query type"""
    config = HealthcareDomainConfig()
    
    if query_type == "emergency":
        return config.SYSTEM_PROMPTS["emergency_triage"]
    elif query_type == "medication":
        return config.SYSTEM_PROMPTS["medication_safety"]
    else:
        return config.SYSTEM_PROMPTS["healthcare_assistant"]

def format_healthcare_response(context, query_type, query_text):
    """Format response with appropriate healthcare disclaimers and structure"""
    config = HealthcareDomainConfig()
    modifiers = config.RESPONSE_MODIFIERS.get(query_type, {})
    
    # Base response with context
    response = context
    
    # Add appropriate disclaimers and warnings
    if query_type == "emergency":
        response += "\n\n🚨 **If this is a medical emergency, call 911 immediately.**"
    
    if modifiers.get("include_disclaimer", True):
        response += "\n\n**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice."
    
    if modifiers.get("include_when_to_see_doctor"):
        response += "\n\n**When to see a doctor:** If symptoms persist, worsen, or you have concerns about your health."
    
    if modifiers.get("emphasize_prescription_compliance"):
        response += "\n\n**Medication Safety:** Always follow your healthcare provider's instructions exactly."
    
    if modifiers.get("emphasize_regular_checkups"):
        response += "\n\n**Regular Care:** Maintain regular checkups with your healthcare provider for ongoing management."
    
    return response

# Example usage in enhanced_rag_voice_agent.py modification
def create_healthcare_enhanced_agent():
    """Example of how to modify the agent for healthcare domain"""
    
    # This would be integrated into your existing enhanced_rag_voice_agent.py
    healthcare_system_message = """You are a Healthcare Information Assistant powered by a comprehensive medical knowledge base.

Your role is to provide accurate, evidence-based health information while maintaining appropriate medical disclaimers and safety guidelines.

CORE PRINCIPLES:
1. Provide factual, evidence-based health information
2. Always include appropriate medical disclaimers
3. Encourage professional medical consultation
4. Identify emergency situations requiring immediate care
5. Use clear, understandable language for health education

SAFETY GUIDELINES:
- Never diagnose medical conditions
- Never prescribe treatments or medications
- Always recommend consulting healthcare professionals
- Emphasize emergency care when appropriate
- Focus on education and general guidance

Remember: You are an educational tool, not a replacement for professional medical care."""
    
    return healthcare_system_message

if __name__ == "__main__":
    # Test the configuration
    test_queries = [
        "I have chest pain and trouble breathing",  # emergency
        "What are the symptoms of diabetes?",       # symptoms
        "How should I take my blood pressure medication?",  # medication
        "How can I prevent heart disease?"          # prevention
    ]
    
    print("🧪 Testing Healthcare Domain Configuration")
    print("=" * 50)
    
    for query in test_queries:
        query_type = classify_healthcare_query(query)
        system_prompt = get_healthcare_system_prompt(query_type)
        
        print(f"\n❓ Query: {query}")
        print(f"📋 Classified as: {query_type}")
        print(f"🎯 System prompt: {system_prompt[:100]}...")
        print("-" * 30)
