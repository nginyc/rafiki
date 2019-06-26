import React from 'react';
import ReactDOM from 'react-dom';
import { render } from '@testing-library/react'
import Datasets from './Datasets';
import { MemoryRouter } from 'react-router';

const Wrapper = (
    <MemoryRouter initialEntries={[ '/Datasets' ]}>
        <Datasets/>
    </MemoryRouter>
)

it('renders without crashing', () => {
    const div = document.createElement('div');
    ReactDOM.render(Wrapper, div);
});

it('renders datasets message', () => {
    const app = render(Wrapper);
    app.getAllByText("Dataset",{exact:false})
});
