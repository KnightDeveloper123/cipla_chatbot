const Joi = require('joi');

const addUserSchema = Joi.object({
    name: Joi.string().required().max(255).messages({
        'string.base': 'Name must be a string.',
        'any.required': 'Name is required'
    }),

    // mobile_no: Joi.string().required().messages({
    //     'string.email': 'Email must be a valid email address.',
    //     'any.email': 'Name is required'
    // }),

    email: Joi.string().required().email().messages({
        'string.email': 'Email must be a valid email address.',
        'any.email': 'Name is required'
    }),
    password: Joi.string().min(4).max(50).required().messages({
        "string.min": "Password must be at least 4 characters long.",
        "string.max": "Password must not exceed 50 characters.",
        "any.required": "Password is required."
    }),

    date_of_birth: Joi.date().iso().less('now').messages({
        'date.base': 'Date of Birth must be a valid date.',
        'date.format': 'Date of Birth must be in YYYY-MM-DD format.',
        'date.less': 'Date of Birth must be in the past.'
    })
});

const updateUserSchema = Joi.object({
    name: Joi.string().optional().max(255).allow(null, '').messages({
        'string.base': 'Name must be a string.',
    }),

    email: Joi.string().optional().email().allow(null, '').messages({
        'string.email': 'Email must be a valid email address.',
    }),
    mobile_no: Joi.string().optional().messages({
        'string.email': 'Email must be a valid email address.',
        'any.email': 'Name is required'
    }),
    password: Joi.string().min(4).max(50).required().messages({
        "string.min": "Password must be at least 4 characters long.",
        "string.max": "Password must not exceed 50 characters.",
        "any.required": "Password is required."
    }),

    date_of_birth: Joi.date().iso().less('now').optional().allow(null, '').messages({
        'date.base': 'Date of Birth must be a valid date.',
        'date.format': 'Date of Birth must be in YYYY-MM-DD format.',
        'date.less': 'Date of Birth must be in the past.'
    }),

    user_id: Joi.number().required().messages({
        'number.base': "User should be int",
        'any.required': 'User is required'
    }),
    status: Joi.number().optional().allow(null, '').messages({
        'number.base': 'User Status is invalid.',
    }),
});

const deleteUserSchema = Joi.object({
    user_id: Joi.number().required().messages({
        'number.base': 'Invalid User',
        'any.required': 'User is required'
    })
})

module.exports = { addUserSchema, updateUserSchema, deleteUserSchema };
