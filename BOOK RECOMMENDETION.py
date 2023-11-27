from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

# Load the dataset (replace 'path/to/your/dataset.csv' with your dataset)
reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5))
data = Dataset.load_from_file('path/to/your/dataset.csv', reader=reader)

# Use a user-based collaborative filtering approach with KNN
sim_options = {
    'name': 'cosine',
    'user_based': True
}

knn_model = KNNBasic(sim_options=sim_options)

# Cross-validate the model
cross_validate(knn_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model on the entire dataset
trainset = data.build_full_trainset()
knn_model.fit(trainset)

# Get book recommendations for a specific user (replace 'user_id' with the actual user ID)
user_id = 'user_id'
user_books = set(trainset.ur[trainset.to_inner_uid(user_id)])

# Exclude books the user has already rated
candidates = list(book_id for book_id in trainset.all_items() if book_id not in user_books)

# Get the top-N book recommendations for the user
k = 10
user_ratings = [(book_id, knn_model.predict(user_id, book_id).est) for book_id in candidates]
top_n = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]

print(f"Top {k} Book Recommendations for User {user_id}:")
for book_id, estimated_rating in top_n:
    print(f"Book ID: {book_id}, Estimated Rating: {estimated_rating}")
