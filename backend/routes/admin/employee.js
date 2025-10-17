const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const { addEmployeeSchema, updateEmployeeSchema, deleteEmployeeSchema } = require("../../validation/employee");
const { middleware } = require('../../middleware/middleware');


router.post("/add", middleware, async (req, res) => {
    try {
        const { name, email, mobile_no, date_of_birth } = req.body;
        const { admin_id } = req.query;
        console.log(req.query,req.body);
        

        const [checkEmail] = await executeQuery(`select * from employee where email=?`, [email])
        if (checkEmail) {
            return res.status(400).json({ error: "Email already exist" })
        }

        const insertQuery = 'insert into employee (name, email, mobile_no, date_of_birth, admin_id) values (?, ?, ?, ?, ?);'
        connection.execute(insertQuery, [name, email, mobile_no, date_of_birth, admin_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            return res.json({ success: "Employee Added", data })
        })
    } catch (error) {
        console.log("/add: ", error.message);
        return res.status(500).json({ error: "Internal Server Error." });
    }
});

router.post("/update", middleware, async (req, res) => {
    try {
        const { employee_id, ...rest } = req.body;
        console.log(req.body);
        
        const { error } = updateEmployeeSchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        const fields = Object.keys(rest);
        const values = Object.values(rest);

        if (fields.length === 0) {
            return res.status(400).json({ error: "No valid fields provided for update." });
        }

        const setClause = fields.map(field => `${field} = ?`).join(", ");
        values.push(employee_id);

        const query = `UPDATE employee SET ${setClause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, values, (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "Employee updated", data })
        });
    } catch (error) {
        console.error("Error in /update :", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.post("/delete", middleware, async (req, res) => {
    try {
        const { user_id } = req.body;

        const { error } = deleteEmployeeSchema.validate(req.body, { abortEarly: false });
        if (error) {
            return res.status(400).json({ error: error.details.map(err => err.message) });
        }

        const query = `UPDATE employee SET status=1, updated_at = CURRENT_TIMESTAMP WHERE id = ?`;

        connection.execute(query, [user_id], (err, data) => {
            if (err) {
                console.log(err);
                return res.status(400).json({ error: "Something went wrong" })
            }
            if (data.affectedRows === 0) {
                return res.status(404).json({ error: "Record not found" });
            }
            return res.json({ success: "User deleted", data })
        });
    } catch (error) {
        console.error("Error in /update:", error.message);
        return res.status(500).json({ error: "Internal Server Error" });
    }
});

router.get('/getEmployeeId', middleware, async (req, res) => {
    try {
        const { user_id } = req.query;
        const data = await executeQuery(`select * from employee where id=${user_id}`)
        return res.json({ data: data[0] })
    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal Server Error" })
    }
});
router.get('/getAllEmployee', middleware, async (req, res) => {
    try {
        const { admin_id } = req.query;
        const data = await executeQuery(`select * from employee where status=0 AND admin_id=${admin_id}`)
        // console.log(data);
        
        return res.json({ success: "success", data })
    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal Server Error" })
    }
});

router.get('/getAllAssignBots', middleware, async (req,res)=>{
    try{
        const {employee_id} =req.query
        const data=await executeQuery(`SELECT 
            sector.id AS sector_id,
            sector.name AS sector_name,
            employee.id AS employee_id,
            employee.name AS employee_name,
            bots.id AS bot_id,
            bots.name AS bot_name,
            bots.nodes ,
            bots.created_at as createdAt,
            bots.edges
            FROM sector
            LEFT JOIN employee ON sector.employee_id = employee.id
            LEFT JOIN bots ON bots.sector_id = sector.id
            WHERE sector.employee_id=${employee_id} `)
        return res.json({success:"success", data})
    } catch(error){
        console.log(error)
        return res.status(500).json({error: "Internal Server Error"})
    }
})

router.get('/getAllAssignGenerativeBots', middleware, async (req,res)=>{
    try{
        const {employee_id} =req.query
        const data=await executeQuery(`SELECT 
            sector.id AS sector_id,
            sector.name AS sector_name,
            employee.id AS employee_id,
            employee.name AS employee_name,
            documents.id AS documents_id,
            documents.created_at as createdAt,
            documents.name
            FROM sector
            LEFT JOIN employee ON sector.employee_id = employee.id
            LEFT JOIN documents ON documents.sector_id = sector.id
            WHERE sector.employee_id=${employee_id} `)
        return res.json({success:"success", data})
    } catch(error){
        console.log(error)
        return res.status(500).json({error: "Internal Server Error"})
    }
})

router.get('/getAllEmployeeQUery', middleware, async (req, res) => {
    try {
        const { admin_id } =req.query;
        const data = await executeQuery(`select * from employee where status=0 and admin_id=${admin_id} `)
      
        return res.json({ success: "success", data })
    } catch (error) {
        console.log(error)
        return res.status(500).json({ error: "Internal Server Error" })
    }
});



module.exports=router