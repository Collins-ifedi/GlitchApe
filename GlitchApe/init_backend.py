import os
import django
import sys
from django.core.management import call_command
from django.contrib.auth import get_user_model
from dotenv import load_dotenv

def main():
    # Load .env if present (Render also provides env vars automatically)
    load_dotenv()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

    print("📦 Running makemigrations...")
    call_command("makemigrations", interactive=False, verbosity=1)

    print("📦 Applying migrations...")
    call_command("migrate", interactive=False, verbosity=1)

    # Superuser credentials from environment
    User = get_user_model()
    admin_username = os.getenv("DJANGO_SUPERUSER_USERNAME", "admin")
    admin_email = os.getenv("DJANGO_SUPERUSER_EMAIL", "admin@glitchape.com")
    admin_password = os.getenv("DJANGO_SUPERUSER_PASSWORD", "admin1234")

    if not User.objects.filter(username=admin_username).exists():
        print(f"👤 Creating superuser {admin_username}...")
        User.objects.create_superuser(
            username=admin_username,
            email=admin_email,
            password=admin_password,
        )
    else:
        print(f"✅ Superuser {admin_username} already exists.")

    print("📦 Collecting static files...")
    call_command("collectstatic", interactive=False, verbosity=1, noinput=True)

    print("🎉 Backend initialization complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)