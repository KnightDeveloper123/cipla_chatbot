const Joi = require('joi');

const addEmployeeSchema = Joi.object({
    name: Joi.string().pattern(/^[A-Za-z\s]+$/).required().max(255).messages({
        'string.base': 'Name must be a string.',
        'string.pattern.base': 'Name must contain only letters and spaces.',
        'any.required': 'Name is required'
    }),
 
    mobile_no: Joi.string()
    .pattern(/^\d{10}$/)
    .required()
    .messages({
        'string.pattern.base': 'Phone number must be exactly 10 digits.',
        'any.required': 'Phone number is required.'
    }),

    email: Joi.string().required().email().messages({
        'string.email': 'Email must be a valid email address.',
        'any.email': 'Name is required'
    }),

    role: Joi.string().required().messages({
        'string.base': 'Role must be a string.',
        'any.role': 'Role is required'
    }),

    date_of_birth: Joi.date().iso().less('now').messages({
        'date.base': 'Date of Birth must be a valid date.',
        'date.format': 'Date of Birth must be in YYYY-MM-DD format.',
        'date.less': 'Date of Birth must be in the past.'
    })
});

const updateEmployeeSchema = Joi.object({
    name: Joi.string().pattern(/^[A-Za-z\s]+$/).required().max(255).messages({
        'string.base': 'Name must be a string.',
        'string.pattern.base': 'Name must contain only letters and spaces.',
        'any.required': 'Name is required'
    }),
   

    email: Joi.string().optional().email().allow(null, '').messages({
        'string.email': 'Email must be a valid email address.',
    }),
    mobile_no: Joi.string()
    .pattern(/^\d{10}$/) 
    .required()
    .messages({
        'string.pattern.base': 'Phone number must be exactly 10 digits.',
        'any.required': 'Phone number is required.'
    }),


    role: Joi.string().optional().allow(null, '').messages({
        'string.base': 'Role must be a string.',
    }),

    date_of_birth: Joi.date().iso().less('now').optional().allow(null, '').messages({
        'date.base': 'Date of Birth must be a valid date.',
        'date.format': 'Date of Birth must be in YYYY-MM-DD format.',
        'date.less': 'Date of Birth must be in the past.'
    }),

    employee_id: Joi.number().required().messages({
        'number.base': "Employee should be int",
        'any.required': 'Employee is required'
    }),
    status: Joi.number().optional().allow(null, '').messages({
        'number.base': 'Employee Status is invalid.',
    }),
});

const deleteEmployeeSchema = Joi.object({
    user_id: Joi.number().required().messages({
        'number.base': 'Invalid User',
        'any.required': 'User is required'
    })
})

module.exports = { addEmployeeSchema, updateEmployeeSchema, deleteEmployeeSchema };
