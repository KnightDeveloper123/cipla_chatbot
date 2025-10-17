const jwt = require("jsonwebtoken");

const middleware = (req, res, next) => {
    try {
        const token =
            req.body.token ||
            req.header("Authorization")

        if (!token || token === undefined) {
            return res.status(401).json({
                success: false,
                message: "Token Missing",
            });
        }
        try {
            const decode = jwt.verify(token, process.env.JWT_SECRET);
            req.user = decode;
        } catch (err) {
            console.log(err);
            return res.status(401).json({
                success: false,
                message: "token is invalid",
            });
        }

        next();
    } catch (err) {
        console.log(err);
        return res.status(401).json({
            success: false,
            message: "Something went wrong while verifying token",
        });
    }
};

module.exports = { middleware };