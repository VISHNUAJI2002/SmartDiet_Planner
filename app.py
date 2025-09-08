from flask import Flask, render_template, request, redirect, url_for, flash
from flask_pymongo import PyMongo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import joblib
import numpy as np
import tensorflow as tf

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/helio")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

mongo = PyMongo(app)

# ✅ Load trained model + preprocessing
model = tf.keras.models.load_model("model/diet_model.keras")
scaler = joblib.load("model/scaler.pkl")
meal_encoder = joblib.load("model/meal_encoder.pkl")

login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.username = user_data["username"]
        self.password_hash = user_data["password"]

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if mongo.db.users.find_one({"username": username}):
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))
        hashed_pw = generate_password_hash(password)
        mongo.db.users.insert_one({"username": username, "password": hashed_pw})
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user_data = mongo.db.users.find_one({"username": username})
        if user_data and check_password_hash(user_data["password"], password):
            user = User(user_data)
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    # Inside app.py dashboard route
    user_likes = mongo.db.likes.find({"user_id": current_user.id})

    pipeline = [
        {"$group": {"_id": "$diet_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    global_likes = list(mongo.db.likes.aggregate(pipeline))

    return render_template("dashboard.html", username=current_user.username, user_likes=user_likes, global_likes=global_likes)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        age = request.form.get("age")
        gender = request.form.get("gender")
        height_cm = request.form.get("height_cm")
        weight_kg = request.form.get("weight_kg")

        # Auto calculate BMI
        bmi = request.form.get("bmi")  # allow manual override
        if height_cm and weight_kg:
            try:
                height_m = float(height_cm) / 100
                weight = float(weight_kg)
                bmi = round(weight / (height_m ** 2), 1)
            except:
                bmi = None

        data = {
            "user_id": current_user.id,
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "chronic_disease": request.form.get("chronic_disease"),
            "blood_pressure_systolic": request.form.get("blood_pressure_systolic"),
            "blood_pressure_diastolic": request.form.get("blood_pressure_diastolic"),
            "cholesterol_level": request.form.get("cholesterol_level"),
            "blood_sugar_level": request.form.get("blood_sugar_level"),
            "sleep_hours": request.form.get("sleep_hours"),
        }

        mongo.db.profiles.update_one({"user_id": current_user.id}, {"$set": data}, upsert=True)
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    profile = mongo.db.profiles.find_one({"user_id": current_user.id})
    return render_template("profile.html", profile=profile)



@app.route("/predict")
@login_required
def predict():
    profile = mongo.db.profiles.find_one({"user_id": current_user.id})

    if not profile:
        flash("Please complete your profile first!", "danger")
        return redirect(url_for("profile"))

    # required fields
    required_fields = ["age", "height_cm", "weight_kg"]
    for field in required_fields:
        if not profile.get(field):
            flash("Age, Height, and Weight are required for prediction!", "danger")
            return redirect(url_for("profile"))

    # build features
    features = [
        float(profile.get("age", 0)),
        float(profile.get("gender", 0)),
        float(profile.get("height_cm", 0)),
        float(profile.get("weight_kg", 0)),
        float(profile.get("bmi", 0)),
        float(profile.get("chronic_disease", 0)),
        float(profile.get("blood_pressure_systolic", 0)),
        float(profile.get("blood_pressure_diastolic", 0)),
        float(profile.get("cholesterol_level", 0)),
        float(profile.get("blood_sugar_level", 0)),
        float(profile.get("sleep_hours", 0)),
    ]

    # predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    meal_idx = int(np.argmax(prediction))

    # mapping index -> diet name
    meal_plan_map = {
        0: "Balanced Diet",
        1: "High Protein",
        2: "Low Carb Diet",
        3: "Low Fat Diet",
        4: "Mediterranean"
    }
    meal_plan = meal_plan_map.get(meal_idx, "Unknown Plan")

    # description dictionary
    meal_plan_descriptions = {
        "Balanced Diet": "Provides a proportionate mix of macronutrients (carbs, proteins, fats) and micronutrients for overall health.",
        "High Protein": "Prioritizes protein intake to support muscle repair, satiety, and metabolic function.",
        "Low Carb Diet": "Reduces carbohydrate intake significantly, encouraging the body to use fat as the primary fuel source.",
        "Low Fat Diet": "Limits total fat intake, particularly saturated fats, to support cardiac health and weight management.",
        "Mediterranean": "Inspired by Mediterranean eating habits, rich in vegetables, olive oil, whole grains, and lean proteins — great for heart health."
    }

    # meal plan options (two example 3-meal options per diet)
    meal_plan_details = {
        "Balanced Diet": [
            {
                "Breakfast": "Oats with nuts and banana",
                "Lunch": "Salad with grilled chicken and whole-wheat bread",
                "Dinner": "Grilled fish with sautéed vegetables and quinoa"
            },
            {
                "Breakfast": "Whole-grain toast with avocado and poached egg",
                "Lunch": "Quinoa bowl with black beans, corn, and roasted veggies",
                "Dinner": "Stir-fried tofu with broccoli, bell peppers, and brown rice"
            }
        ],
        "High Protein": [
            {
                "Breakfast": "Scrambled eggs with spinach and Greek yogurt",
                "Lunch": "Grilled chicken with lentil soup and broccoli",
                "Dinner": "Lean steak or baked tofu with mixed vegetables"
            },
            {
                "Breakfast": "Cottage cheese with almonds, blueberries, hemp seeds",
                "Lunch": "Turkey lettuce wraps with hummus and sliced veggies",
                "Dinner": "Baked lemon-herb cod with quinoa salad and edamame"
            }
        ],
        "Low Carb Diet": [
            {
                "Breakfast": "Avocado and egg bowl with chia seeds",
                "Lunch": "Spinach salad with chicken, feta, olives, olive oil",
                "Dinner": "Pan-seared salmon with kale and mushrooms"
            },
            {
                "Breakfast": "Spinach & feta omelet with sautéed mushrooms",
                "Lunch": "Zucchini noodles with pesto, shrimp, sun-dried tomatoes",
                "Dinner": "Roast chicken thighs with cauliflower mash & green beans"
            }
        ],
        "Low Fat Diet": [
            {
                "Breakfast": "Oatmeal with skim milk, berries, banana",
                "Lunch": "Lentil vegetable soup with whole-grain crackers",
                "Dinner": "Steamed chicken breast with brown rice and veggies"
            },
            {
                "Breakfast": "Smoothie with skim milk, banana, mango, protein powder",
                "Lunch": "Baked potato with fat-free Greek yogurt and broccoli",
                "Dinner": "Shrimp and veggie skewers with seasoned brown rice"
            }
        ],
        "Mediterranean": [
            {
                "Breakfast": "Greek yogurt with walnuts, honey, fresh figs",
                "Lunch": "Chickpea & quinoa salad with cucumber, tomatoes, feta",
                "Dinner": "Baked salmon with roasted asparagus and whole-wheat pita"
            },
            {
                "Breakfast": "Whole-wheat pita with hummus, cucumbers, olives",
                "Lunch": "Farro salad with olives, tomatoes, feta, lemon dressing",
                "Dinner": "Grilled lamb chops with roasted eggplant & zucchini"
            }
        ]
    }

    # get recommendations (MUST do this BEFORE counting likes per plan)
    recommendations = meal_plan_details.get(meal_plan, [])
    description = meal_plan_descriptions.get(meal_plan, "A healthy meal plan recommendation.")

    # likes per plan (plan-level counts) and whether current user liked each plan
    likes_per_plan = []
    user_liked_per_plan = []
    for i, plan in enumerate(recommendations, start=1):
        count = mongo.db.likes.count_documents({
            "diet_name": meal_plan,
            "plan_index": i
        })
        likes_per_plan.append(count)

        # check if current user already liked this specific plan
        existing = mongo.db.likes.find_one({
            "user_id": current_user.id,
            "diet_name": meal_plan,
            "plan_index": i
        })
        user_liked_per_plan.append(existing is not None)

    # total likes for the whole diet (optional)
    total_likes = mongo.db.likes.count_documents({"diet_name": meal_plan})

    return render_template(
        "prediction.html",
        meal=meal_plan,
        description=description,
        recommendations=recommendations,
        likes_per_plan=likes_per_plan,
        user_liked_per_plan=user_liked_per_plan,
        total_likes=total_likes
    )



@app.route("/like/<diet_name>/<int:plan_index>")
@login_required
def like_diet(diet_name, plan_index):
    existing_like = mongo.db.likes.find_one({
        "user_id": current_user.id,
        "diet_name": diet_name,
        "plan_index": plan_index
    })

    if not existing_like:
        mongo.db.likes.insert_one({
            "user_id": current_user.id,
            "diet_name": diet_name,
            "plan_index": plan_index
        })

    flash(f"You liked {diet_name} - Diet {plan_index}", "success")
    return redirect(url_for("predict"))






@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
