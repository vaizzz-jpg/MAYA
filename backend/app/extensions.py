"""Central Flask extension instances.

Extensions are created here and initialized in the application factory
to avoid circular imports across blueprints, models, and services.
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
