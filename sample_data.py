# Sample hotel information for testing with metadata
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Our hotel gym features state-of-the-art equipment including: Treadmills, ellipticals, stationary bikes,
        Free weights and weight machines, Yoga and stretching area. Note: Personal trainers available.

        Gym Hours: the gym is open from 8:00 AM to 11:00 PM daily (except Sundays: from 8:00 AM to 5:00 PM )

        Gym Location: 2nd floor

        Gym Reservation: Not required for gym use, but needed for personal training
        """,
        "metadata": {
            "category": "Gym"
        }
    },
    {
        "content": """
        The spa offers a range of treatments: Swedish and deep tissue massage, Facials and skin treatments, 
        Manicure and pedicure, Aromatherapy.

        For couples, we offer two packages:
        HoneyMoon package: -20 % on all treatments
        Love4Ever package: -30 % on all treatments

        Spa Pricing: $50 for basic services and $150 for premium services.

        Spa Hours: 9am-8pm daily

        Spa Location: 3rd floor

        Spa Reservation: 24-hour notice is required
        """,
        "metadata": {
            "category": "Spa"
        }
    },
    {
        "content": """
        Our signature restaurant features:
        International cuisine with local specialties
        Breakfast buffet: 6:30am-10:30am
        Lunch: 12pm-3pm
        Dinner: 6pm-11pm
        Room service available 24/7

        Resaturant Location: Ground floor

        Restaurant Discounts: 15 % off any meal reservation made for groups larger than 5 people
        """,
        "metadata": {
            "category": "Restaurant"
        }
    }
]
