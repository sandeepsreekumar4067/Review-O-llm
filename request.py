import requests
restaurant_reviews = {
    "restaurant_name": "The Gourmet Spot",
    "reviews": [
        {
            "review_id": 1,
            "customer_name": "John Doe",
            "rating": 5,
            "review_text": "The food was exceptional! Great ambiance and the staff were very attentive. Will definitely come back!",
            "date": "2023-09-12",
        },
        {
            "review_id": 2,
            "customer_name": "Jane Smith",
            "rating": 4,
            "review_text": "Loved the appetizers, but the main course was a bit too salty for my taste. Overall a good experience.",
            "date": "2023-09-14",
        },
        {
            "review_id": 3,
            "customer_name": "Sam Wilson",
            "rating": 3,
            "review_text": "The service was slow, but the food was decent. Nothing too special.",
            "date": "2023-09-18",
        },
        {
            "review_id": 4,
            "customer_name": "Emily Davis",
            "rating": 5,
            "review_text": "Best dining experience I've had in a while! The chef's special was mind-blowing. Highly recommended!",
            "date": "2023-09-21",
        },
        {
            "review_id": 5,
            "customer_name": "Mark Lee",
            "rating": 2,
            "review_text": "Disappointed. The food was cold when served, and the portions were smaller than expected.",
            "date": "2023-09-22",
        },
    ],
}
response = requests.post('http://127.0.0.1:8000/ai',json=restaurant_reviews)
print(response.status_code)
print(response.json())