import React from 'react';
import ReactDOM from 'react-dom';
import { render } from '@testing-library/react'
import App from '../App';


describe("App" , function() {
    it('renders without crashing', () => {
        const div = document.createElement('div');
        ReactDOM.render(<App />, div);
    });

    it('renders login page', () => {
        const app = render(<App />);
        app.getAllByText("Login")
        expect(app.container).toMatchSnapshot()
    });
})