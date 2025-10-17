const express = require('express');
const router = express.Router();
const connection = require('../../database/db');
const executeQuery = require('../../utils/executeQuery');
const fs = require('fs');
const { middleware } = require('../../middleware/middleware');
const multer = require('multer')
const path = require('path');
const { Await } = require('react-router-dom');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, path.join(__dirname, '../../products'));
    },
    filename: (req, file, cb) => {
        const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1E9)}${path.extname(file.originalname)}`;
        cb(null, uniqueName);
    },
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 },
});

router.post('/add', middleware,upload.single('image'), async (req, res) => {
    try {
        const { name, display_name, description, admin_id, sector_id, price, cta } = req.body;
        console.log("req", req.body)

        if (!req.file) {
            return res.status(400).json({ error: 'Image file is required.' });
        }

        const file_name = req.file.filename;
        console.log("file",file_name)

        const query = 'INSERT INTO product_service (name,display_name, description, image, admin_id, sector_id,price,cta) VALUES (?, ?, ?, ?, ?,?,?,?)';
        connection.query(query, [name, display_name, description, file_name, admin_id, sector_id, price, cta], (err, data) => {
            if (err) {
                console.error(err);
                return res.status(400).json({ error: "Something went wrong" });
            }
            console.log("data", data)
            return res.status(200).json({
                success: 'File uploaded successfully.',
                data
            });
        });

    } catch (error) {
        console.error('/add Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});

router.post('/update', middleware, upload.single('image'), async (req, res) => {
    try {
        const { name, display_name, description, product_id, sector_id, price, cta } = req.body;

        let query = 'UPDATE product_service SET name = ?, display_name=?,description = ?, sector_id= ?,price=?,cta=?';
        const values = [name, display_name, description, sector_id, price, cta];

        if (req.file) {
            const file_name = req.file.filename;
            query += ', image = ?';
            values.push(file_name);
        }

        query += ' WHERE id = ?';
        values.push(product_id);

        connection.query(query, values, (err, result) => {
            if (err) {
                console.error(err);
                return res.status(400).json({ error: 'Update failed.' });
            }

            return res.status(200).json({
                success: 'Product updated successfully.',
                result
            });
        });

    } catch (error) {
        console.error('/update Error:', error.message);
        return res.status(500).json({ error: 'Internal Server Error.' });
    }
});




router.post('/delete_product', middleware, (req, res) => {
    try {
        const { product_id } = req.body
        connection.query(`update product_service set status='1', updated_on=NOW() where id=?`, [product_id], (err, data) => {
            if (err) {
                console.log(err);
                res.status(400).json({ error: "Something went wrong!" })
            } else {
                res.json({ data, success: "Product Deleted!" })
            }
        })
    } catch (error) {
        console.log(error);
        res.status(500).json({ error: "Internal server error!" })
    }
})

router.get('/allDeletedProduct', async (req, res) => {
    try {
        const data = await executeQuery(`select * from product_service where status=1`)

        return res.json({ data, success: "Product Deleted!" })

    } catch (error) {
        console.log(error);
        res.status(500).json({ error: "Internal server error!" })
    }
})

// GET BY PRODUCT

router.get('/product_by_id', middleware, async (req, res) => {
    try {
        const { product_id } = req.query
        const product = await executeQuery(`select  * from product_service where id=${product_id}`)
        return res.json({ product: product[0], })
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error!" })
    }
})

// GET ALL PRODUCT
router.get('/get_all_product', middleware, async (req, res) => {
    try {
        const { admin_id } = req.query
        const product = await executeQuery(`SELECT ps.*, sector.name AS sector_name
            from  
            product_service as ps
            left join sector on ps.sector_id=sector.id
             where  ps.status = 0 AND ps.admin_id=${admin_id} order by ps.id desc;`)
        return res.json({ product, })
    } catch (error) {
        console.log(error);
        return res.status(500).json({ error: "Internal server error!" })
    }
})


router.get('/get_all_sector_bots', async (req, res) => {
    try {
        const { product_id } = req.query
        const product = await executeQuery(`select product_sector.product_id ,product_sector.sector_id,sector.name as s_name,bots.name,bots.id,bots.nodes,bots.created_at as createdAt from product_sector
            left join sector on  product_sector.sector_id =sector.id
           left join bots on bots.sector_id=sector.id
            where product_id=${product_id}`)
        return res.json({ product })
    } catch (error) {

    }
})

router.get('/get_all_sector', async (req, res) => {
    try {
        const { product_id } = req.query
        const product = await executeQuery(`select product_sector.product_id ,product_sector.sector_id,sector.name as s_name,
            sector.icon , sector.category,sector.description  from product_sector
            left join sector on  product_sector.sector_id =sector.id
         where product_id=${product_id}`)
        return res.json({ product })
    } catch (error) {

    }
})




module.exports = router;