# DevConnector

DevConnector is a full-stack social networking platform for developers, built using the MERN (MongoDB, Express.js, React, Node.js) stack. It allows developers to create profiles, connect with others, share posts, and engage in discussions.

## Features

- User authentication and authorization using JWT
- Profile creation with work experience and education details
- Ability to add social media links to the profile
- Post creation, liking, and commenting functionalities
- Developer profiles browsing and searching
- Backend API built with Node.js and Express
- MongoDB as the database with Mongoose ODM
- Frontend developed using React with Redux for state management

## Technologies Used

### Frontend:

- React.js
- Redux
- Axios
- Bootstrap

### Backend:

- Node.js
- Express.js
- MongoDB & Mongoose
- JSON Web Token (JWT)
- bcrypt.js for password hashing

## Installation

### Prerequisites:

- Node.js installed
- MongoDB installed and running

### Steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/devconnector.git
   cd devconnector
   ```

2. Install dependencies:

   ```sh
   npm install
   cd client
   npm install
   ```

3. Set up environment variables: Create a `.env` file in the root directory and add:

   ```env
   MONGO_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   ```

4. Start the development server:

   ```sh
   npm run dev
   ```

   This will run both the frontend and backend concurrently.

## Usage

- Register an account and log in.
- Create and edit your developer profile.
- View other developers' profiles and connect with them.
- Share posts, like, and comment on them.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

If you have any questions or issues, feel free to reach out via GitHub issues or contact me at [[your-email@example.com](mailto\:your-email@example.com)].

