import numpy as np
import psycopg2
import json
import uuid
import sys
# --- CONFIGURATION ---
person_id = sys.argv[1] if len(sys.argv) > 1 else ""
NPY_FILE_PATH = f"data/vectors/{person_id}.npy"

# Mapping of angle keys to (is_frontal, is_primary)
ANGLE_CONFIG = {
    'center': (True, True),       # frontal, primary encoding
    'look_up': (False, False),
    'look_down': (False, False),
    'left_semi': (False, False),
    'right_semi': (False, False),
    'left_full': (False, False),
    'right_full': (False, False),
}
DB_CONFIG = {
    "dbname": "workflow_system",
    "user": "postgres",
    "password": "9ets0n1234",
    "host": "10.26.1.175",
    "port": "5432"
}

def migrate_embeddings():
    # 1. Load the NPY file
    print(f"Loading {NPY_FILE_PATH}...")
    try:
        # allow_pickle=True is needed if the NPY contains a dictionary/objects
        data = np.load(NPY_FILE_PATH, allow_pickle=True)

        # Handle different NPY structures
        if isinstance(data, np.ndarray) and data.ndim == 0:
            # It's a 0-d array wrapping a dictionary (common in numpy saves)
            embeddings_dict = data.item()
        elif isinstance(data, dict):
            embeddings_dict = data
        else:
            print("Error: NPY file format not recognized. Expected a dictionary of {angle_name: vector}.")
            return

        # Extract person_id from filename (e.g., "NH3775.npy" -> "NH3775")
        # person_id = NPY_FILE_PATH.split('/')[-1].replace('.npy', '')

        # Load person name from metadata file
        metadata_path = f"data/metadata/{person_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        person_name = metadata['name']

        print(f"Found {len(embeddings_dict)} angle vectors for person {person_id} ({person_name}).")

    except Exception as e:
        print(f"Failed to load NPY file: {e}")
        return

    # 2. Connect to Database
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        insert_query = """
        INSERT INTO workflow_runtime.iot_face_encodings
        (face_encoding_id, person_type, encoding_vector, encoding_model, encoding_dimension,
        face_location, face_quality_score, face_angle, is_frontal, is_primary,
        tenant_id, is_active, person_id, person_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """

        count = 14
        for angle_name, vector in embeddings_dict.items():
            # Ensure vector is a standard Python list (Postgres doesn't understand numpy arrays directly)
            if hasattr(vector, 'tolist'):
                vector_data = vector.tolist()
            else:
                vector_data = vector
            safe_vector = vector_data  # Use this if column is ARRAY or VECTOR type

            # Get angle config (is_frontal, is_primary) or default to (False, False)
            is_frontal, is_primary = ANGLE_CONFIG.get(angle_name, (False, False))
            dimension = len(vector_data) if isinstance(vector_data, list) else 512

            cur.execute(insert_query, (
                str(uuid.uuid4()),   # face_encoding_id
                'employee',          # person_type
                safe_vector,         # encoding_vector
                'insightface',       # encoding_model
                dimension,           # encoding_dimension
                None,                # face_location (null for base encodings)
                None,                # face_quality_score
                json.dumps({'angle':angle_name}),          # face_angle (e.g., 'center', 'look_up', etc.)
                is_frontal,          # is_frontal
                is_primary,          # is_primary
                'T689',              # tenant_id (adjust as needed)
                True,                # is_active
                person_id,           # person_id
                person_name          # person_name from metadata
            ))
            count += 1

        conn.commit()
        print(f"Successfully inserted {count} records for {person_id} into workflow_runtime.iot_face_encodings.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database Error: {error}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    migrate_embeddings()