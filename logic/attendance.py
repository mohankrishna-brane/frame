import numpy as np
from collections import Counter

class RecognitionSystem:
    def __init__(self, db_adapter, threshold=0.45):
        # Load the entire database into RAM for speed
        # Structure: { "EMP001": { "name": "John", "vectors": [v1, v2...] } }
        self.db = db_adapter.load_identities()
        self.threshold = threshold
        
        # Buffer to store votes: { track_id: ["EMP001", "EMP001", "Unknown", ...] }
        self.vote_buffer = {} 
        self.BUFFER_SIZE = 8  # Require 8 frames of history
        self.CONSENSUS_REQ = 5 # At least 5 must match

    def identify(self, track_id, live_vector):
        """
        1. Find best match for this vector.
        2. Add to voting buffer for this track_id.
        3. Check if we have a consensus.
        """
        # A. Vector Search (1:N Match)
        best_id = "Unknown"
        best_score = 0.0
        
        # Normalize live vector
        live_vector = live_vector / np.linalg.norm(live_vector)

        # Brute-force search (Fine for <1000 users. For >1000, use FAISS)
        for emp_id, data in self.db.items():
            for stored_vec in data['vectors']:
                score = np.dot(live_vector, stored_vec) # Cosine Similarity
                if score > best_score:
                    best_score = score
                    best_id = emp_id

        # Threshold Check
        final_decision = "Unknown"
        if best_score > self.threshold:
            final_decision = best_id

        # B. Voting Logic
        if track_id not in self.vote_buffer:
            self.vote_buffer[track_id] = []
        
        buffer = self.vote_buffer[track_id]
        buffer.append(final_decision)
        
        # Keep buffer size fixed
        if len(buffer) > self.BUFFER_SIZE:
            buffer.pop(0)

        # C. Consensus Check
        # Count votes in the buffer
        counts = Counter(buffer)
        most_common_id, votes = counts.most_common(1)[0]
        
        # We only return a NAME if:
        # 1. It is not "Unknown"
        # 2. It has enough votes
        if most_common_id != "Unknown" and votes >= self.CONSENSUS_REQ:
            name = self.db[most_common_id]['name']
            return most_common_id, name, best_score
        
        return "Unknown", "Unknown", best_score