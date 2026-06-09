const router = require('express').Router();
const { users } = require('../store');
const { signToken } = require('../middleware/auth');

/**
 * POST /api/auth/login
 * Body: { username, password }
 * Returns: { token, user }
 */
router.post('/login', (req, res) => {
  const { username, password } = req.body || {};
  if (!username || !password)
    return res.status(400).json({ error: 'username and password required' });

  const user = users.get(username);
  if (!user || user.password !== password)
    return res.status(401).json({ error: 'Invalid credentials' });

  const token = signToken(user);
  res.json({ token, user: { id: user.id, username: user.username, role: user.role } });
});

/**
 * GET /api/auth/me  (requires auth — tested in main app)
 */
router.get('/me', require('../middleware/auth').requireAuth, (req, res) => {
  res.json({ user: req.user });
});

module.exports = router;
