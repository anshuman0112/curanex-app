"""Builds the exercise catalog used by the app (src/data/exercises.json) and the API (backend/api/data/exercises.json).
Edit the entries here, then run:  python backend/tools/build_exercise_catalog.py
Thresholds are joint angles in degrees from MediaPipe Pose; see src/lib/exerciseEngine.ts for metric definitions."""
import json, sys
from pathlib import Path
E = []
def lt(v): return {"lt": v}
def gt(v): return {"gt": v}
def rule(metric, test, issue, cue, phase="any"): return {"metric": metric, "test": test, "issue": issue, "cue": cue, "phase": phase}
def tgt(metric, lo, hi, cue): return {"metric": metric, "min": lo, "max": hi, "cue": cue}
def add(id, name, cat, *, equip="none", level="beginner", impact="low", muscles=(), view="side", met=4.0, mode="reps",
        count=None, rules=(), hold=None, sets=3, reps=None, seconds=None, rng=None, avoid=(), caution=(), steps=(), camera=True, show=(), source=None):
    d = {"id": id, "name": name, "category": cat, "equipment": equip, "level": level, "impact": impact, "muscles": list(muscles),
         "view": view, "met": met, "mode": mode, "camera": camera, "steps": list(steps), "avoid": sorted(set(avoid)), "caution": sorted(set(caution) - set(avoid)),
         "defaults": {"sets": sets, **({"reps": reps} if reps else {}), **({"seconds": seconds} if seconds else {})}}
    d["range"] = rng or ({"sets": [1, 5], "reps": [4, 25]} if mode == "reps" else {"sets": [1, 4], "seconds": [10, 90] if mode == "hold" else [20, 600]})
    if count: d["count"] = count
    if rules: d["rules"] = list(rules)
    if hold: d["hold"] = hold
    if show: d["show"] = [list(s) for s in show]
    if source: d["source"] = source
    assert (mode == "reps") == bool(reps), id
    assert mode == "reps" or seconds, id
    assert mode != "hold" or hold, id
    E.append(d)

REPO = "ai-gym-coach (Shradha Khapra)"
HIGH = ["knee", "heart", "pregnancy", "osteoporosis", "senior", "obesity_severe"]
CHEST_UP = rule("torsoLean", gt(50), "The user is leaning too far forward.", "Keep your chest up, you are leaning too far forward.", "in")
STRAIGHT = rule("body:best", lt(150), "The user's body is not straight.", "Keep your body in one straight line from head to heels.")
SAG = rule("hipSag", gt(0.3), "The user's hips are sagging.", "Your hips are sagging. Tighten your core and lift them.")
PIKE = rule("hipSag", lt(-0.3), "The user's hips are too high.", "Hips too high. Lower them into a straight line.")

# ---------------- Bodyweight strength ----------------
add("squat", "Bodyweight squat", "bodyweight", muscles=["quads", "glutes"], met=5.0, reps=12,
    count={"metric": "knee:best", "enter": lt(100), "exit": gt(160), "partial": {"at": 140, "cue": "Go lower. Sit your hips back until your thighs are near parallel."}},
    rules=[CHEST_UP], caution=["knee"], source=REPO, show=[["knee:best", "Knee angle"], ["torsoLean", "Torso lean"]],
    steps=["Feet shoulder-width apart, toes slightly out.", "Push hips back and bend knees until thighs are near parallel.", "Chest up, knees in line with toes.", "Drive through your heels to stand fully."])
add("sumo-squat", "Sumo squat", "bodyweight", view="front", muscles=["inner thighs", "glutes"], met=5.0, reps=12,
    count={"metric": "knee:min", "enter": lt(110), "exit": gt(160), "partial": {"at": 140, "cue": "Sink a little lower, keep knees pushing out."}},
    rules=[rule("footSpread", lt(1.5), "The user's stance is too narrow for a sumo squat.", "Step your feet wider than your shoulders.")], caution=["knee"],
    steps=["Face the camera, feet wide, toes turned out.", "Bend knees and lower straight down, knees tracking over toes.", "Keep your torso upright.", "Press up through your heels."])
add("chair-sit-to-stand", "Chair sit-to-stand", "low_impact", equip="chair", muscles=["quads", "glutes"], met=3.5, reps=10,
    count={"metric": "knee:best", "enter": lt(105), "exit": gt(160)}, show=[["knee:best", "Knee angle"]],
    steps=["Sit at the front of a sturdy chair, feet flat, side-on to the camera.", "Lean slightly forward and stand up fully.", "Slowly sit back down with control.", "Use your hands on your thighs if you need help."])
add("wall-sit", "Wall sit", "bodyweight", equip="wall", mode="hold", muscles=["quads"], met=4.0, sets=2, seconds=30,
    hold={"targets": [tgt("knee:best", 70, 120, "Slide down until your knees are near a right angle."), tgt("hip:best", 70, 125, "Keep your back flat against the wall.")]},
    avoid=["knee", "heart", "hypertension"], steps=["Back flat against a wall, side-on to the camera.", "Slide down until knees are about 90 degrees.", "Knees above ankles.", "Breathe steadily, do not hold your breath."])
add("lunge", "Forward lunge", "bodyweight", muscles=["quads", "glutes"], met=5.0, reps=10,
    count={"metric": "knee:min", "enter": lt(100), "exit": gt(160), "partial": {"at": 135, "cue": "Drop your back knee lower toward the floor."}},
    rules=[rule("torsoLean", gt(30), "The user is leaning forward in the lunge.", "Keep your torso upright over your hips.", "in")],
    avoid=["knee", "balance"], caution=["senior"], source=REPO, show=[["knee:min", "Front knee"], ["torsoLean", "Torso lean"]],
    steps=["Stand tall side-on, then step one foot forward.", "Lower until both knees are near 90 degrees.", "Front knee stays over the ankle.", "Push back to standing and switch legs."])
add("reverse-lunge", "Reverse lunge", "bodyweight", muscles=["quads", "glutes"], met=5.0, reps=10,
    count={"metric": "knee:min", "enter": lt(100), "exit": gt(160), "partial": {"at": 135, "cue": "Step back further and lower your back knee."}},
    rules=[rule("torsoLean", gt(30), "The user is leaning forward.", "Stay tall through your chest.", "in")], avoid=["knee", "balance"],
    steps=["Stand tall side-on to the camera.", "Step one foot back and lower both knees.", "Keep weight in the front heel.", "Return to standing and alternate."])
add("side-lunge", "Side lunge", "bodyweight", view="front", muscles=["inner thighs", "glutes"], met=5.0, reps=10,
    count={"metric": "knee:min", "enter": lt(120), "exit": gt(160)},
    rules=[rule("torsoLean", gt(35), "The user is folding forward in the side lunge.", "Lift your chest, sit back into the bent leg.", "in")], avoid=["knee", "balance"],
    steps=["Face the camera, feet wide.", "Shift to one side, bending that knee and sitting back.", "Keep the other leg straight.", "Push back to centre and switch sides."])
add("glute-bridge", "Glute bridge", "bodyweight", equip="mat", muscles=["glutes", "hamstrings"], met=3.5, reps=12,
    count={"metric": "hip:best", "enter": gt(160), "exit": lt(140)},
    rules=[rule("knee:best", gt(125), "The user's feet are too far from the hips.", "Bring your feet closer to your hips.")], caution=["pregnancy"],
    show=[["hip:best", "Hip angle"]],
    steps=["Lie on your back side-on to the camera, knees bent, feet flat.", "Press through heels and lift hips until your body is straight from knees to shoulders.", "Squeeze glutes for one second.", "Lower slowly."])
add("pushup", "Push-up", "bodyweight", equip="mat", level="intermediate", muscles=["chest", "triceps", "core"], met=6.0, reps=10,
    count={"metric": "elbow:best", "enter": lt(90), "exit": gt(160), "partial": {"at": 125, "cue": "Lower your chest further, elbows to about 90 degrees."}},
    rules=[STRAIGHT, SAG, PIKE], caution=["shoulder", "wrist"], avoid=["pregnancy"], source=REPO,
    show=[["elbow:best", "Elbow angle"], ["body:best", "Body line"]],
    steps=["Side-on to the camera, hands under shoulders, body in one straight line.", "Lower until elbows reach about 90 degrees.", "Keep hips level.", "Push back up to straight arms."])
add("knee-pushup", "Knee push-up", "bodyweight", equip="mat", muscles=["chest", "triceps"], met=4.0, reps=10,
    count={"metric": "elbow:best", "enter": lt(95), "exit": gt(155), "partial": {"at": 125, "cue": "Go a little lower with your chest."}},
    rules=[rule("hip:best", lt(145), "The user is bending at the hips.", "Keep a straight line from knees to shoulders.")], caution=["shoulder", "wrist"],
    steps=["Side-on, knees on the mat, hands under shoulders.", "Straight line from knees to head.", "Lower chest toward the floor.", "Push back up."])
add("wall-pushup", "Wall push-up", "low_impact", equip="wall", muscles=["chest", "triceps"], met=3.0, reps=10,
    count={"metric": "elbow:best", "enter": lt(110), "exit": gt(155)}, rules=[rule("body:best", lt(150), "The user's body is bent.", "Keep your body straight from head to heels.")],
    caution=["shoulder"], steps=["Stand an arm's length from a wall, side-on to the camera.", "Hands on the wall at shoulder height.", "Bend elbows to bring your chest toward the wall.", "Push away, breathing out."])
add("incline-pushup", "Incline push-up", "bodyweight", equip="chair", muscles=["chest", "triceps"], met=4.5, reps=10,
    count={"metric": "elbow:best", "enter": lt(100), "exit": gt(155)}, rules=[STRAIGHT, SAG], caution=["shoulder", "wrist"],
    steps=["Side-on, hands on a sturdy bench or chair seat.", "Body straight from head to heels.", "Lower chest to the edge.", "Push back up."])
add("chair-dip", "Chair triceps dip", "bodyweight", equip="chair", level="intermediate", muscles=["triceps"], met=4.5, reps=10,
    count={"metric": "elbow:best", "enter": lt(100), "exit": gt(150)}, avoid=["shoulder", "wrist"],
    steps=["Side-on, hands on the edge of a fixed chair behind you.", "Lower by bending elbows to about 90 degrees.", "Keep your back close to the chair.", "Press back up."])

# ---------------- Gym strength (dumbbells) ----------------
add("goblet-squat", "Goblet squat", "strength", equip="dumbbells", level="intermediate", muscles=["quads", "glutes"], met=5.5, reps=10,
    count={"metric": "knee:best", "enter": lt(100), "exit": gt(160), "partial": {"at": 140, "cue": "Go deeper, elbows inside your knees."}},
    rules=[CHEST_UP], caution=["knee", "back"], steps=["Side-on, hold one dumbbell at your chest.", "Squat down, keeping chest tall.", "Knees track over toes.", "Drive up through heels."])
add("dumbbell-curl", "Biceps curl (dumbbell)", "strength", equip="dumbbells", muscles=["biceps"], met=3.5, reps=12,
    count={"metric": "elbow:best", "enter": lt(50), "exit": gt(160)},
    rules=[rule("elbowDrift", gt(0.2), "The user's elbow is drifting away from their side.", "Keep your elbow pinned to your side."),
           rule("torsoLean", gt(15), "The user is swinging the torso.", "Stop swinging. Keep your body still and let the arm do the work.")],
    source=REPO, show=[["elbow:best", "Elbow angle"], ["torsoLean", "Torso swing"]],
    steps=["Stand tall side-on, elbows tucked at your sides.", "Curl the weight up until the elbow is fully bent.", "Lower slowly to straight arms.", "Don't swing your torso."])
add("hammer-curl", "Hammer curl", "strength", equip="dumbbells", muscles=["biceps", "forearms"], met=3.5, reps=12,
    count={"metric": "elbow:best", "enter": lt(55), "exit": gt(160)},
    rules=[rule("elbowDrift", gt(0.2), "The user's elbow is drifting.", "Keep your elbows close to your body."),
           rule("torsoLean", gt(15), "The user is swinging.", "Keep your torso still.")],
    steps=["Side-on, palms facing each other.", "Curl up keeping elbows at your sides.", "Pause at the top.", "Lower with control."])
add("shoulder-press", "Shoulder press", "strength", equip="dumbbells", muscles=["shoulders", "triceps"], met=4.0, reps=10,
    count={"metric": "elbow:best", "enter": gt(160), "exit": lt(90)},
    rules=[rule("hip:best", lt(150), "The user is arching their lower back.", "Don't arch your lower back. Brace your core.")],
    avoid=["shoulder"], caution=["hypertension", "back"], source=REPO, show=[["elbow:best", "Elbow angle"], ["hip:best", "Back line"]],
    steps=["Side-on, weights at shoulder height, elbows bent.", "Press straight up to full extension.", "Lower back to shoulder height.", "Brace your core, ribs down."])
add("lateral-raise", "Lateral raise", "strength", equip="dumbbells", view="front", muscles=["shoulders"], met=3.5, reps=12,
    count={"metric": "armRaise:min", "enter": gt(75), "exit": lt(30)},
    rules=[rule("armRaise:max", gt(115), "The user is raising the arms above shoulder height.", "Stop at shoulder height."),
           rule("torsoLean", gt(15), "The user is swinging.", "Keep your body still, lift with your shoulders.")],
    caution=["shoulder"], steps=["Face the camera, weights by your sides.", "Raise both arms out to shoulder height.", "Slight bend in the elbows.", "Lower slowly."])
add("front-raise", "Front raise", "strength", equip="dumbbells", muscles=["shoulders"], met=3.5, reps=12,
    count={"metric": "armRaise:best", "enter": gt(75), "exit": lt(30)},
    rules=[rule("torsoLean", gt(15), "The user is leaning back.", "Stand tall, don't lean back to lift.")], caution=["shoulder"],
    steps=["Stand side-on, weights in front of thighs.", "Raise arms straight forward to shoulder height.", "Pause briefly.", "Lower with control."])
add("bent-over-row", "Bent-over row", "strength", equip="dumbbells", level="intermediate", muscles=["back", "biceps"], met=4.5, reps=10,
    count={"metric": "elbow:best", "enter": lt(95), "exit": gt(150)},
    rules=[rule("torsoLean", lt(25), "The user is too upright for a row.", "Hinge forward from your hips with a flat back.", "in")], caution=["back"],
    steps=["Side-on, hinge forward at the hips, back flat.", "Let the weights hang below your shoulders.", "Pull elbows back toward your hips.", "Lower with control."])
add("romanian-deadlift", "Romanian deadlift", "strength", equip="dumbbells", level="intermediate", muscles=["hamstrings", "glutes", "back"], met=5.0, reps=10,
    count={"metric": "hip:best", "enter": lt(110), "exit": gt(165)},
    rules=[rule("knee:best", lt(130), "The user is squatting instead of hinging.", "Soft knees. Push your hips back, not down.", "in")], caution=["back"],
    steps=["Stand tall side-on holding the weights.", "Push hips back, sliding weights down your thighs.", "Keep your back flat and knees soft.", "Squeeze glutes to stand up."])
add("overhead-triceps", "Overhead triceps extension", "strength", equip="dumbbells", muscles=["triceps"], met=3.5, reps=12,
    count={"metric": "elbow:best", "enter": lt(70), "exit": gt(150)},
    rules=[rule("shoulder:best", lt(140), "The user's elbows are dropping forward.", "Keep your elbows pointing up by your ears.")], avoid=["shoulder"],
    steps=["Side-on, hold one weight overhead with both hands.", "Lower it behind your head by bending elbows.", "Keep elbows pointing up.", "Extend back to straight arms."])
add("good-morning", "Good morning", "strength", level="intermediate", muscles=["hamstrings", "back"], met=4.0, reps=12,
    count={"metric": "hip:best", "enter": lt(115), "exit": gt(165)}, avoid=["back", "osteoporosis"],
    steps=["Side-on, hands behind your head, feet hip-width.", "Hinge forward with a flat back.", "Stop when you feel the hamstrings stretch.", "Return to standing."])

# ---------------- Core ----------------
add("plank", "Forearm plank", "core", equip="mat", mode="hold", muscles=["core"], met=4.0, sets=3, seconds=30,
    hold={"targets": [tgt("body:best", 155, 180, "Straighten your body from head to heels."), tgt("hipSag", -0.3, 0.3, "Keep your hips level with your shoulders."), tgt("elbow:best", 60, 125, "Rest on your forearms, elbows under shoulders.")]},
    caution=["hypertension", "shoulder"], avoid=["pregnancy", "heart"],
    steps=["Side-on, forearms on the mat, elbows under shoulders.", "Legs straight, body in one line.", "Tighten your core and glutes.", "Breathe steadily."])
add("high-plank", "High plank", "core", equip="mat", mode="hold", muscles=["core", "shoulders"], met=4.0, sets=3, seconds=30,
    hold={"targets": [tgt("body:best", 155, 180, "Straighten your body."), tgt("hipSag", -0.3, 0.3, "Keep your hips level."), tgt("elbow:best", 150, 180, "Straighten your arms.")]},
    caution=["wrist", "hypertension"], avoid=["pregnancy", "heart"],
    steps=["Side-on, hands under shoulders, arms straight.", "Body in one straight line.", "Brace your core.", "Breathe steadily."])
add("side-plank", "Side plank", "core", equip="mat", level="intermediate", mode="hold", view="front", muscles=["obliques"], met=4.0, sets=2, seconds=20,
    hold={"targets": [tgt("body:best", 150, 180, "Lift your hips so your body is straight."), tgt("torsoLean", 45, 90, "Turn onto your side.")]},
    caution=["shoulder", "hypertension"], avoid=["pregnancy"],
    steps=["Lie on your side facing the camera, elbow under shoulder.", "Lift hips so your body is straight.", "Stack your feet.", "Hold and breathe."])
add("situp", "Sit-up", "core", equip="mat", level="intermediate", muscles=["abs"], met=4.0, reps=12,
    count={"metric": "hip:best", "enter": lt(70), "exit": gt(110)}, avoid=["back", "osteoporosis", "pregnancy"],
    steps=["Lie on your back, knees bent, side-on to the camera.", "Curl up until your chest nears your thighs.", "Lower with control.", "Don't pull on your neck."])
add("crunch", "Crunch", "core", equip="mat", muscles=["abs"], met=3.5, reps=15,
    count={"metric": "hip:best", "enter": lt(105), "exit": gt(120)}, avoid=["osteoporosis", "pregnancy"], caution=["back"],
    steps=["Lie on your back side-on, knees bent.", "Lift shoulders off the floor using your abs.", "Keep your chin off your chest.", "Lower slowly."])
add("leg-raise", "Lying leg raise", "core", equip="mat", level="intermediate", muscles=["lower abs"], met=4.0, reps=10,
    count={"metric": "body:best", "enter": lt(120), "exit": gt(160)},
    rules=[rule("knee:best", lt(140), "The user is bending the knees.", "Keep your legs as straight as you can.", "in")], avoid=["back", "pregnancy"],
    steps=["Lie flat, legs straight, side-on to the camera.", "Lift legs up toward the ceiling.", "Lower slowly without arching your back.", "Stop just above the floor."])
add("bird-dog", "Bird dog", "core", equip="mat", muscles=["core", "back"], met=3.0, reps=10,
    count={"metric": "armRaise:max", "enter": gt(145), "exit": lt(110)},
    rules=[rule("hipLevel", gt(0.25), "The user's hips are tilting.", "Keep your hips level.")],
    steps=["On hands and knees, side-on to the camera.", "Reach one arm forward and the opposite leg back.", "Keep hips level and back flat.", "Return and switch sides."])
add("mountain-climbers", "Mountain climbers", "cardio", equip="mat", level="intermediate", impact="high", mode="timed", muscles=["core", "full body"], met=8.0, sets=3, seconds=30,
    count={"metric": "hip:min", "enter": lt(100), "exit": gt(140)}, rules=[SAG, PIKE], avoid=HIGH + ["wrist"], caution=["shoulder"],
    steps=["Side-on, start in a high plank.", "Drive one knee toward your chest.", "Switch legs quickly.", "Keep hips low and level."])

# ---------------- Cardio / weight loss ----------------
add("jumping-jacks", "Jumping jacks", "cardio", view="front", impact="high", mode="timed", muscles=["full body"], met=8.0, sets=3, seconds=40,
    count={"metric": "armRaise:min", "enter": gt(140), "exit": lt(45)},
    rules=[rule("footSpread", lt(1.3), "The user's feet are not going wide.", "Jump your feet out wide as your arms go up.", "in")], avoid=HIGH, caution=["asthma", "hypertension"],
    steps=["Face the camera, feet together.", "Jump feet wide while raising arms overhead.", "Jump back to start.", "Land softly."])
add("step-jacks", "Step jacks (low impact)", "low_impact", view="front", mode="timed", muscles=["full body"], met=4.5, sets=3, seconds=40,
    count={"metric": "armRaise:min", "enter": gt(130), "exit": lt(45)},
    steps=["Face the camera.", "Step one foot out while raising arms overhead.", "Step back in and lower arms.", "Alternate sides."])
add("high-knees", "High knees", "cardio", impact="high", mode="timed", muscles=["legs", "core"], met=8.0, sets=3, seconds=30,
    count={"metric": "hip:min", "enter": lt(110), "exit": gt(155)}, avoid=HIGH, caution=["asthma", "hypertension"],
    steps=["Stand side-on to the camera.", "Run in place driving knees to hip height.", "Pump your arms.", "Stay light on your feet."])
add("butt-kicks", "Butt kicks", "cardio", impact="high", mode="timed", muscles=["hamstrings"], met=7.0, sets=3, seconds=30,
    count={"metric": "knee:min", "enter": lt(70), "exit": gt(150)}, avoid=HIGH, caution=["asthma"],
    steps=["Stand side-on.", "Jog in place kicking heels toward your glutes.", "Keep your chest up.", "Stay light and quick."])
add("jump-squat", "Jump squat", "cardio", level="intermediate", impact="high", muscles=["quads", "glutes"], met=8.0, reps=10,
    count={"metric": "knee:best", "enter": lt(110), "exit": gt(165)}, rules=[CHEST_UP], avoid=HIGH + ["hypertension", "back"],
    steps=["Side-on, squat down with chest up.", "Explode up into a jump.", "Land softly with bent knees.", "Go straight into the next squat."])
add("burpee", "Burpee", "cardio", level="advanced", impact="high", muscles=["full body"], met=10.0, reps=8,
    count={"metric": "torsoLean", "enter": gt(65), "exit": lt(20)}, avoid=HIGH + ["hypertension", "back", "wrist", "shoulder", "asthma"],
    steps=["Side-on, from standing squat and place hands on the floor.", "Jump feet back to a plank.", "Jump feet back in.", "Stand or jump up."])
add("shadow-boxing", "Shadow boxing", "cardio", view="front", mode="timed", muscles=["shoulders", "arms", "core"], met=6.0, sets=3, seconds=45,
    count={"metric": "elbow:max", "enter": gt(155), "exit": lt(100)}, caution=["shoulder"],
    steps=["Face the camera in a boxing stance, hands up by your chin.", "Punch straight out, then pull back to guard.", "Alternate arms and keep moving your feet.", "Breathe out on each punch."])
add("march-in-place", "March in place", "low_impact", mode="timed", muscles=["legs"], met=3.5, sets=1, seconds=60,
    count={"metric": "hip:min", "enter": lt(150), "exit": gt(168)},
    steps=["Stand side-on to the camera.", "Lift knees gently one at a time.", "Swing your arms.", "Keep a steady pace."])
add("walk-in-place", "Brisk walk in place", "low_impact", mode="timed", muscles=["legs"], met=3.5, sets=1, seconds=180,
    steps=["Stand where the camera can see you.", "Walk briskly in place.", "Swing your arms naturally.", "Breathe through your nose."])
add("torso-twist", "Standing torso twist", "mobility", view="front", mode="timed", muscles=["core", "spine"], met=2.5, sets=1, seconds=40,
    caution=["back", "osteoporosis", "pregnancy"], steps=["Face the camera, feet hip-width apart, arms bent at chest height.", "Turn your upper body gently left and right.", "Keep hips facing forward.", "Move slowly, breathe out as you turn."])

# ---------------- Low impact / seniors ----------------
add("seated-knee-extension", "Seated knee extension", "low_impact", equip="chair", muscles=["quads"], met=2.5, reps=10,
    count={"metric": "knee:best", "enter": gt(155), "exit": lt(115)},
    steps=["Sit tall on a chair, side-on to the camera.", "Straighten one knee until the leg is level.", "Hold for a second.", "Lower slowly and switch legs."])
add("seated-march", "Seated march", "low_impact", equip="chair", mode="timed", muscles=["hip flexors"], met=2.5, sets=2, seconds=45,
    count={"metric": "hip:best", "enter": lt(78), "exit": gt(88)},
    steps=["Sit tall on a chair, side-on to the camera.", "Lift one knee, then lower it.", "Alternate legs.", "Keep your back upright."])
add("side-leg-raise", "Standing side leg raise", "low_impact", view="front", muscles=["hips", "glutes"], met=3.0, reps=10,
    count={"metric": "legSpread", "enter": gt(30), "exit": lt(12)},
    rules=[rule("torsoLean", gt(15), "The user is leaning sideways.", "Stay tall, don't lean to lift your leg.")], caution=["balance"],
    steps=["Face the camera, hold a chair for balance if needed.", "Lift one leg out to the side.", "Keep toes pointing forward.", "Lower slowly and switch."])
add("standing-knee-raise", "Standing knee raise", "low_impact", muscles=["hip flexors", "core"], met=3.0, reps=10,
    count={"metric": "hip:min", "enter": lt(120), "exit": gt(160)}, caution=["balance"],
    steps=["Stand side-on, hold a chair if needed.", "Lift one knee toward hip height.", "Lower slowly.", "Alternate legs."])

# ---------------- Yoga ----------------
Y = "yoga"
add("tadasana", "Mountain pose (Tadasana)", Y, view="front", mode="hold", muscles=["posture"], met=2.0, sets=1, seconds=30,
    hold={"targets": [tgt("knee:min", 165, 180, "Straighten both legs."), tgt("torsoLean", 0, 10, "Stand tall, stack shoulders over hips.")]},
    steps=["Stand facing the camera, feet together.", "Weight even on both feet.", "Lengthen your spine, shoulders relaxed.", "Breathe slowly."])
add("vrikshasana", "Tree pose (Vrikshasana)", Y, view="front", mode="hold", muscles=["balance", "legs"], met=2.5, sets=2, seconds=30,
    hold={"targets": [tgt("knee:max", 160, 180, "Keep your standing leg straight."), tgt("knee:min", 0, 115, "Place your foot on the inner thigh or calf, knee out to the side."),
                      tgt("torsoLean", 0, 12, "Stand tall, don't lean.")]}, avoid=["balance"], caution=["senior", "pregnancy"],
    steps=["Face the camera, stand on one leg.", "Place the other foot on your inner calf or thigh (never the knee).", "Hands together at the chest or overhead.", "Fix your gaze on one point and breathe."])
add("virabhadrasana-2", "Warrior II (Virabhadrasana II)", Y, view="front", mode="hold", muscles=["legs", "shoulders"], met=3.0, sets=2, seconds=30,
    hold={"targets": [tgt("footSpread", 2.0, 8, "Step your feet wider apart."), tgt("knee:min", 80, 130, "Bend your front knee toward 90 degrees."),
                      tgt("knee:max", 155, 180, "Keep your back leg straight."), tgt("armRaise:min", 65, 115, "Stretch both arms out at shoulder height.")]},
    caution=["knee"], steps=["Face the camera, feet wide.", "Turn one foot out and bend that knee over the ankle.", "Arms out at shoulder height.", "Gaze over the front hand, breathe."])
add("virabhadrasana-1", "Warrior I (Virabhadrasana I)", Y, mode="hold", muscles=["legs", "hips"], met=3.0, sets=2, seconds=30,
    hold={"targets": [tgt("knee:min", 80, 130, "Bend your front knee more."), tgt("knee:max", 150, 180, "Straighten your back leg."), tgt("wristsUp", 1, 1, "Reach both arms overhead.")]},
    caution=["knee", "shoulder", "hypertension"], steps=["Side-on to the camera, step one foot back.", "Bend the front knee over the ankle.", "Reach both arms overhead.", "Square your hips and breathe."])
add("virabhadrasana-3", "Warrior III (Virabhadrasana III)", Y, level="advanced", mode="hold", muscles=["balance", "hamstrings"], met=3.0, sets=2, seconds=15,
    hold={"targets": [tgt("torsoLean", 60, 90, "Tip your torso forward until it is level."), tgt("knee:max", 155, 180, "Straighten the standing leg.")]},
    avoid=["balance", "senior", "pregnancy", "hypertension"], steps=["Stand side-on, shift weight to one leg.", "Hinge forward, lifting the other leg behind you.", "Body forms a T.", "Hold and breathe."])
add("utkatasana", "Chair pose (Utkatasana)", Y, mode="hold", muscles=["quads", "glutes"], met=3.0, sets=2, seconds=30,
    hold={"targets": [tgt("knee:best", 90, 150, "Bend your knees more, as if sitting on a chair."), tgt("armRaise:best", 130, 180, "Raise your arms alongside your ears.")]},
    caution=["knee"], steps=["Side-on, feet together.", "Bend knees and sit hips back.", "Raise arms overhead.", "Keep weight in your heels and breathe."])
add("adho-mukha-svanasana", "Downward dog (Adho Mukha Svanasana)", Y, equip="mat", mode="hold", muscles=["hamstrings", "shoulders"], met=3.0, sets=2, seconds=30,
    hold={"targets": [tgt("hip:best", 40, 110, "Lift your hips up and back."), tgt("elbow:best", 145, 180, "Straighten your arms."), tgt("knee:best", 140, 180, "Straighten your legs as much as you can.")]},
    avoid=["hypertension", "pregnancy"], caution=["wrist", "shoulder"],
    steps=["Side-on, start on hands and knees.", "Lift hips up to an upside-down V.", "Press hands into the mat, heels toward the floor.", "Breathe slowly."])
add("bhujangasana", "Cobra pose (Bhujangasana)", Y, equip="mat", mode="hold", muscles=["back"], met=2.5, sets=2, seconds=20,
    hold={"targets": [tgt("torsoLean", 35, 90, "Lie on your belly, side-on to the camera."), tgt("hip:best", 115, 170, "Gently lift your chest using your back.")]},
    avoid=["pregnancy"], caution=["back"], steps=["Lie on your belly, side-on, hands under shoulders.", "Press gently to lift your chest.", "Keep hips on the floor, elbows bent.", "Breathe and lower slowly."])
add("phalakasana", "Plank pose (Phalakasana)", Y, equip="mat", mode="hold", muscles=["core"], met=3.5, sets=2, seconds=20,
    hold={"targets": [tgt("body:best", 155, 180, "Straighten your body."), tgt("hipSag", -0.3, 0.3, "Hips in line with shoulders."), tgt("elbow:best", 150, 180, "Straighten your arms.")]},
    avoid=["pregnancy", "heart"], caution=["wrist", "hypertension"], steps=["Side-on, hands under shoulders.", "Body in one line.", "Engage your core.", "Breathe evenly."])
add("trikonasana", "Triangle pose (Trikonasana)", Y, view="front", mode="hold", muscles=["hamstrings", "obliques"], met=2.5, sets=2, seconds=30,
    hold={"targets": [tgt("footSpread", 1.8, 8, "Step your feet wider."), tgt("knee:min", 155, 180, "Keep both legs straight."),
                      tgt("torsoLean", 25, 90, "Tilt sideways over your front leg."), tgt("armRaise:max", 130, 180, "Reach your top arm up to the sky.")]},
    caution=["balance", "hypertension"], steps=["Face the camera, feet wide.", "Turn one foot out.", "Tilt sideways, lower hand to shin, top arm up.", "Keep both legs straight and breathe."])
add("setu-bandhasana", "Bridge pose (Setu Bandhasana)", Y, equip="mat", mode="hold", muscles=["glutes", "back"], met=2.5, sets=2, seconds=30,
    hold={"targets": [tgt("hip:best", 150, 180, "Lift your hips higher."), tgt("knee:best", 50, 115, "Feet closer to your hips.")]},
    caution=["pregnancy"], steps=["Lie on your back, side-on, knees bent.", "Lift hips toward the ceiling.", "Arms on the floor by your sides.", "Breathe and hold."])
add("balasana", "Child's pose (Balasana)", Y, equip="mat", mode="hold", muscles=["back", "hips"], met=1.5, sets=1, seconds=45,
    hold={"targets": [tgt("hip:best", 0, 65, "Sit your hips back toward your heels."), tgt("knee:best", 0, 65, "Fold your knees fully.")]},
    caution=["knee", "pregnancy"], steps=["Kneel side-on to the camera.", "Sit hips back to your heels.", "Fold forward, arms extended or by your sides.", "Breathe deeply and relax."])
add("navasana", "Boat pose (Navasana)", Y, equip="mat", level="intermediate", mode="hold", muscles=["core"], met=3.5, sets=2, seconds=20,
    hold={"targets": [tgt("hip:best", 45, 110, "Lean back and lift your legs into a V."), tgt("knee:best", 110, 180, "Straighten your legs more if you can.")]},
    avoid=["back", "pregnancy", "hypertension", "heart"], steps=["Sit side-on to the camera.", "Lean back and lift your feet.", "Arms forward, spine long.", "Breathe and hold."])
add("utkata-konasana", "Goddess pose (Utkata Konasana)", Y, view="front", mode="hold", muscles=["inner thighs", "glutes"], met=3.0, sets=2, seconds=30,
    hold={"targets": [tgt("footSpread", 1.8, 8, "Step your feet wider."), tgt("knee:max", 75, 140, "Bend both knees, sink your hips."), tgt("torsoLean", 0, 15, "Keep your torso upright.")]},
    caution=["knee"], steps=["Face the camera, feet wide, toes out.", "Bend knees and sink hips.", "Knees track over toes.", "Arms in cactus shape, breathe."])
add("paschimottanasana", "Seated forward bend (Paschimottanasana)", Y, equip="mat", mode="hold", muscles=["hamstrings", "back"], met=2.0, sets=1, seconds=40,
    hold={"targets": [tgt("hip:best", 0, 80, "Fold forward from your hips."), tgt("knee:best", 145, 180, "Keep your legs straight, a small bend is fine.")]},
    avoid=["osteoporosis", "pregnancy"], caution=["back"], steps=["Sit side-on, legs straight in front.", "Inhale to lengthen the spine.", "Exhale and fold forward from the hips.", "Hold your shins or feet and breathe."])
add("uttanasana", "Standing forward bend (Uttanasana)", Y, mode="hold", muscles=["hamstrings"], met=2.0, sets=1, seconds=30,
    hold={"targets": [tgt("hip:best", 0, 80, "Fold forward from your hips."), tgt("knee:best", 145, 180, "Keep legs straight, soft knees are fine.")]},
    avoid=["osteoporosis", "hypertension"], caution=["back", "balance"], steps=["Stand side-on, feet hip-width.", "Fold forward from the hips.", "Let your head and arms hang.", "Come up slowly."])
add("side-bend", "Standing side bend", Y, view="front", mode="hold", muscles=["obliques"], met=2.0, sets=2, seconds=20,
    hold={"targets": [tgt("torsoLean", 12, 45, "Lean sideways a little more."), tgt("armRaise:max", 130, 180, "Reach your arm up over your head.")]},
    steps=["Face the camera, feet together.", "Raise one or both arms overhead.", "Bend gently to one side.", "Return and switch sides."])
add("cat-cow", "Cat-cow (Marjaryasana-Bitilasana)", Y, equip="mat", mode="timed", muscles=["spine"], met=2.0, sets=1, seconds=60,
    caution=["wrist"], steps=["On hands and knees, side-on.", "Inhale: drop the belly, lift the chest.", "Exhale: round the back, tuck the chin.", "Move slowly with your breath."])
add("shavasana", "Corpse pose (Shavasana)", Y, equip="mat", mode="timed", camera=False, muscles=["relaxation"], met=1.0, sets=1, seconds=180,
    steps=["Lie on your back, arms by your sides.", "Close your eyes.", "Relax every part of the body.", "Breathe naturally."])
add("sukhasana-breathing", "Deep belly breathing (Sukhasana)", "breathing", mode="timed", camera=False, muscles=["breathing"], met=1.3, sets=1, seconds=180,
    steps=["Sit comfortably, spine tall.", "Breathe in through the nose for 4 counts, belly expands.", "Breathe out slowly for 6 counts.", "Keep shoulders relaxed."])
add("anulom-vilom", "Alternate nostril breathing (Anulom Vilom)", "breathing", mode="timed", camera=False, muscles=["breathing"], met=1.3, sets=1, seconds=180,
    caution=["asthma"], steps=["Sit tall, close the right nostril with your thumb.", "Breathe in through the left.", "Close the left, breathe out through the right.", "Keep alternating gently. No breath-holding."])

# ---------------- Mobility / stretching ----------------
add("quad-stretch", "Standing quad stretch", "mobility", mode="hold", muscles=["quads"], met=2.0, sets=2, seconds=20,
    hold={"targets": [tgt("knee:min", 0, 60, "Pull your heel closer to your glutes."), tgt("knee:max", 160, 180, "Keep the standing leg straight.")]},
    caution=["balance", "knee"], steps=["Stand side-on, hold a chair if needed.", "Bend one knee and hold the ankle behind you.", "Knees close together.", "Hold, then switch."])
add("neck-rolls", "Gentle neck rolls", "mobility", mode="timed", camera=False, muscles=["neck"], met=1.5, sets=1, seconds=40,
    caution=["balance"], steps=["Sit or stand tall.", "Drop chin to chest.", "Roll gently ear to shoulder, both ways.", "Never force the range."])
add("shoulder-rolls", "Shoulder rolls", "mobility", view="front", mode="timed", muscles=["shoulders"], met=1.5, sets=1, seconds=40,
    steps=["Face the camera.", "Roll shoulders up, back and down.", "10 times backward, 10 forward.", "Keep the neck relaxed."])
add("arm-circles", "Arm circles", "mobility", view="front", mode="timed", muscles=["shoulders"], met=2.5, sets=1, seconds=40, caution=["shoulder"],
    steps=["Face the camera, arms out at shoulder height.", "Make small circles forward.", "Gradually make them bigger.", "Reverse direction."])
add("hip-circles", "Hip circles", "mobility", view="front", mode="timed", muscles=["hips"], met=2.0, sets=1, seconds=40,
    steps=["Face the camera, hands on hips, feet hip-width.", "Circle your hips slowly one way.", "Then the other way.", "Keep your upper body still."])

for e in E:  # posture targets first so the cue names the real fault (e.g. sagging hips)
    if e.get("hold"): e["hold"]["targets"].sort(key=lambda t: 0 if t["metric"] in ("hipSag", "footSpread") else 1)
ids = [e["id"] for e in E]; assert len(ids) == len(set(ids))
CONDITIONS = {
    "knee": ["knee", "arthritis", "osteoarthritis", "acl", "meniscus", "patella", "ligament"],
    "back": ["back pain", "backache", "spine", "spinal", "slip disc", "slipped disc", "disc", "sciatica", "lumbar", "spondyl", "scoliosis"],
    "shoulder": ["shoulder", "rotator", "frozen"],
    "wrist": ["wrist", "carpal"],
    "hypertension": ["hypertension", "high blood pressure", "high bp", "blood pressure"],
    "heart": ["heart", "cardiac", "angina", "arrhythmia", "coronary", "bypass", "stent", "cardiomyopathy", "heart failure"],
    "pregnancy": ["pregnan"],
    "balance": ["vertigo", "balance", "parkinson", "dizziness", "dizzy", "stroke", "epilep", "neuropathy"],
    "osteoporosis": ["osteopor", "osteopenia"],
    "asthma": ["asthma", "copd", "bronchitis"],
}
LABELS = {"knee": "knee problems", "back": "back problems", "shoulder": "shoulder problems", "wrist": "wrist problems", "hypertension": "high blood pressure",
          "heart": "a heart condition", "pregnancy": "pregnancy", "balance": "balance problems", "osteoporosis": "osteoporosis", "asthma": "asthma",
          "senior": "age 65+", "obesity_severe": "BMI 35 or more"}
CATEGORIES = {"strength": "Gym strength", "bodyweight": "Bodyweight", "yoga": "Yoga", "cardio": "Cardio & weight loss", "core": "Core",
              "mobility": "Mobility & stretching", "low_impact": "Low-impact & seniors", "breathing": "Breathing"}
out = json.dumps({"version": 1, "categories": CATEGORIES, "conditions": CONDITIONS, "conditionLabels": LABELS, "exercises": E}, indent=1, ensure_ascii=False) + chr(10)
root = Path(__file__).resolve().parents[2]
for p in (root / "src" / "data" / "exercises.json", root / "backend" / "api" / "data" / "exercises.json"):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(out, encoding="utf-8")
print(f"{len(E)} exercises", file=sys.stderr)
