# Vercel Python Serverless Function
# Serves same routes as backend/app.py but via Vercel's python runtime.
# Vercel will route /api/* to this file. Local dev still uses backend/app.py.

from backend.app import app

# Vercel expects `app` to be the Flask app
# (vercel.json rewrites /api/(.*) → /api/index.py)

# For compatibility with Vercel's @vercel/python, expose handler
def handler(request, response):
    return app(request, response)
