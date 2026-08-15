The redraw does not land near the teacher's 1.2913 and it does not move
toward it. So the two lines the review put at risk both stand:

- **A3's student degrades at bb200k.** 1.3618 → 1.3010 → 1.3998, and →
  1.4098 on the second draw. The ladder's +0.0988 is +0.1088 read off draw 2.
- **A3's student/teacher gap is real.** 0.1084 on draw 1, 0.1185 on draw 2,
  against a next-largest of 0.0168 in group A and 0.0425 anywhere. Two head
  seeds put that student above its teacher by ~3x the band, so the gap is a
  property of that student encoder rather than of one head draw.

What this closes, and what it does not. It removes "one bad draw" as the
explanation. It does not explain the gap: A3 is the cell where `k = 3` does
the most damage, and this study has no second backbone seed on it.

**What the redraw held, and what it did not.** Draw 1 trained its head on the
rented box; draw 2 trained on elisa. Both read the same 200,000-step backbone
checkpoint, the box's original and elisa's synced copy of it, and both evals
ran on elisa's cores over the same 97 configs. So the 0.0100 between the two
draws bounds the head seed and the machine together, not the seed alone. That
makes the agreement a stronger result, not a weaker one. The 0.1084
student/teacher gap holds the machine, because draw 1 and the teacher both
trained on the box; the redraw's 0.1185 crosses machines. Only elisa's copy of
the backbone carries a recorded md5
(`9f0e8da71ff595523d2bf0dabdf80445`); the box was released before its original
could be checksummed.
