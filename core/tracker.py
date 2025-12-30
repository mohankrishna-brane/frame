import numpy as np

class SimpleTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.objects = {} # Format: { id: [x1, y1, x2, y2] }
        self.disappeared = {} # Count how many frames an ID has been missing
        self.max_lost = max_lost # How long to wait before forgetting an ID

    def update(self, rects):
        # rects = list of [x1, y1, x2, y2]
        
        # 1. If no faces detected, mark existing objects as 'missing'
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_lost:
                    self.deregister(obj_id)
            return self.objects

        # 2. If we are tracking nothing, register everything we see
        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
        else:
            # 3. Match new boxes (rects) to existing objects using IOU
            # We want to find which new box overlaps most with an old box
            
            new_objects = {} # Keep track of what we matched this frame
            
            # Loop through every new face found in this frame
            for rect in rects:
                best_id = -1
                max_iou = 0.0
                
                # Check against every currently tracked object
                for obj_id, existing_rect in self.objects.items():
                    iou = self._iou(existing_rect, rect)
                    
                    # If this box overlaps more than previous best, keep it
                    if iou > max_iou:
                        max_iou = iou
                        best_id = obj_id
                
                # THRESHOLD: overlap must be > 20% to be considered the "same person"
                if max_iou > 0.2: 
                    # Update the existing object with the new coordinates
                    self.objects[best_id] = rect
                    self.disappeared[best_id] = 0 # Reset missing counter
                    new_objects[best_id] = rect
                else:
                    # No overlap found? It's a new person.
                    self.register(rect)

            # 4. Clean up objects that were NOT matched in this frame
            for obj_id in list(self.objects.keys()):
                if obj_id not in new_objects:
                    # If it wasn't updated, increase its missing count
                    self.disappeared[obj_id] += 1
                    
                    # If missing for too long, delete it
                    if self.disappeared[obj_id] > self.max_lost:
                        self.deregister(obj_id)

        return self.objects

    def register(self, rect):
        self.objects[self.next_id] = rect
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def _iou(self, boxA, boxB):
        # Intersection over Union (Overlap Calculation)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou