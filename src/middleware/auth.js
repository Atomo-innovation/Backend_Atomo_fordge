const jwt = require('jsonwebtoken');

const SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';

function signToken(user) {
  return jwt.sign(
    { id: user.id, username: user.username, role: user.role },
    SECRET,
    { expiresIn: '24h' }
  );
}

function requireAuth(req, res, next) {
  const header = req.headers['authorization'] || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Missing auth token' });

  try {
    req.user = jwt.verify(token, SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid or expired token' });
  }
}

const ROLE_RANK = { viewer: 0, operator: 1, admin: 2 };

function requireRole(minRole) {
  return (req, res, next) => {
    const rank = ROLE_RANK[req.user?.role] ?? -1;
    if (rank < (ROLE_RANK[minRole] ?? 99)) {
      return res.status(403).json({ error: `Requires ${minRole} role or higher` });
    }
    next();
  };
}

module.exports = { signToken, requireAuth, requireRole };
