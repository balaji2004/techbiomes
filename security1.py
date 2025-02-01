from flask import Flask, render_template, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_session import Session

app = Flask(__name__)

# Configure Secret Key & Session Type
app.config["SECRET_KEY"] = "supersecretkey"  # Change in production
app.config["SESSION_TYPE"] = "filesystem"

Session(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Mock user database
users = {"admin": {"password": "password123"}}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user authentication."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username]["password"] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    """Protected route that requires login."""
    return "Welcome to the Dashboard!"

@app.route("/logout")
@login_required
def logout():
    """Log the user out and clear session."""
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
