const Joi = require("joi");

const addQuerySchema = Joi.object({
    query: Joi.string().required().messages({
        "string.base": "Query must be a string.",
        "any.required": "Query is required.",
    }),
    user_id: Joi.number().required().messages({
        "number.base": "User ID must be a number.",
        "any.required": "User ID is required.",
    }),
});

const updateQuerySchema = Joi.object({
    query: Joi.string().optional().allow(null, "").messages({
        "string.base": "Query must be a string.",
    }),
    user_id: Joi.number().required().messages({
        "number.base": "User ID must be a number.",
        "any.required": "User ID is required.",
    }),
    assignee_id: Joi.number().optional().allow(null, "").messages({
        "number.base": "Assignee ID must be a number.",
    }),
    query_status: Joi.string().valid("pending", "resolved", "closed").optional().messages({
        "string.base": "Query status must be a string.",
        "any.only": "Query status must be one of: pending, resolved, closed.",
    }),
    query_id: Joi.number().required().messages({
        'number.base': "Query should be int",
        'any.required': 'Query is required'
    }),
});

module.exports = { addQuerySchema, updateQuerySchema };
